from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext
import inspect

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Scheduler, device_name_enum, model_size_enum, model_name_enum, Optimizations, StepPreviewMode, ImageGenerationResult
from ..models import Pipeline

#from diffusers import StableDiffusionDepth2ImgPipeline
# tokenizer

from pathlib import Path

pipeline = Pipeline.STABLE_DIFFUSION

import diffusers
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from openvino.runtime import Model, Core
from transformers import CLIPTokenizer
import torch
import open_clip
import PIL.Image
import PIL.ImageOps
 


class StableDiffusionEngineDepth(diffusers.DiffusionPipeline):
    def __init__(
        self,
        model="bes-dev/stable-diffusion-v1-4-openvino",
        device="CPU"
     
    ):   
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        print("Starting Depth Model load")

        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        self.text_encoder = self.core.compile_model(os.path.join(model, "text_encoder.xml"), device)
        self._text_encoder_output = self.text_encoder.output(0)

        self.unet = self.core.compile_model(os.path.join(model, "unet.xml"), device)
        self._unet_output = self.unet.output(0)
        self.latent_shape = tuple(self.unet.inputs[0].shape)[1:]
        self.vae_decoder = self.core.compile_model(os.path.join(model, "vae_decoder.xml"), device)
        self.vae_encoder = self.core.compile_model(os.path.join(model, "vae_encoder.xml"), device) 

        self.init_image_shape = tuple(self.vae_encoder.inputs[0].shape)[2:]

        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None

        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8  

        super().__init__()

        self.vae_scale_factor = 8
        self.scaling_factor =  0.18215 
        
        
    def prepare_depth(self, depth, image, dtype, device):
        device = torch.device('cpu') # if device.type == 'mps' else device.type)
       
        if depth is None:
            print("Depth is None so Depth setting up the depth model")
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            import contextlib
      
            
            feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas") 
            depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas") 
            depth_estimator = depth_estimator.to(device)
            
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
         
            pixel_values = pixel_values.to(device=device)
            # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
            # So we use `torch.autocast` here for half precision inference.
            #context_manger = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
            context_manger = contextlib.nullcontext()
            with context_manger:
                depth_map = depth_estimator(pixel_values).predicted_depth
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(self.height // self.vae_scale_factor, self.width // self.vae_scale_factor),
                mode="bicubic",
                align_corners=False,
            )

            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_map = depth_map.to("cpu",dtype) #depth_map.to(device)
            return depth_map
        else:
            if isinstance(depth, PIL.Image.Image):
                depth = np.array(depth.convert("L"))
                depth = depth.astype(np.float32) / 255.0
            depth = depth[None, None]
            depth = torch.from_numpy(depth)
            return depth
            
    def prepare_depth_latents(
        self, depth, batch_size, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        depth = torch.nn.functional.interpolate(
            depth, size=(self.height // self.vae_scale_factor, self.width // self.vae_scale_factor)
        )
        depth = depth.to(device=device,dtype=torch.float)
        #depth = depth.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        depth = depth.repeat(batch_size, 1, 1, 1)
        depth = torch.cat([depth] * 2) if do_classifier_free_guidance else depth
        return depth

    def prepare_img2img_latents(self, batch_size, num_channels_latents, dtype, device, generator, latents=None, image=None, timestep=None,scheduler=LMSDiscreteScheduler):
        shape = (batch_size, num_channels_latents, self.height // self.vae_scale_factor, self.width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" #if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to("cpu")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

        if image is not None:
            image = image.to(device="cpu", dtype=dtype)
            if isinstance(generator, list):
                image_latents = [
                    self.vae.encode(image[0:1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.sample(generator)
            image_latents = torch.nn.functional.interpolate(
                image_latents, size=(self.height // self.vae_scale_factor, self.width // self.vae_scale_factor)
            )
            image_latents = 0.18215 * image_latents
            rand_device = "cpu" #if device.type == "mps" else device
            shape = image_latents.shape
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                noise = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype) for i in
                    range(batch_size)
                ]
                noise = torch.cat(noise, dim=0).to(device)
            else:
                noise = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
            latents = scheduler.add_noise(image_latents, noise, timestep)

        return latents
    def prepare_latents(self, image = None, scheduler=LMSDiscreteScheduler, latent_timestep:torch.Tensor = None):
        """
        Function for getting initial latents for starting generation
        
        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(scheduler, LMSDiscreteScheduler):
                noise = noise * scheduler.sigmas[0].numpy()
                return noise
            elif isinstance(scheduler, EulerDiscreteScheduler):
                noise = noise * scheduler.sigmas.max().numpy()
                return noise
            else:
                return noise
            
        input_image = image
        moments = self.vae_encoder(input_image)[self._vae_e_output]
        mean, logvar = np.split(moments, 2, axis=1) 
        std = np.exp(logvar * 0.5)
        latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta,scheduler):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def get_timesteps(self, num_inference_steps, strength, scheduler):
        # get the original timestep using init_timestep
        offset = scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep + offset , 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        depth_image: Union[torch.FloatTensor, PIL.Image.Image],
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        scheduler=None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        
        # 0. Default height and width to unet
        #height = height or 768 #self.unet.config.sample_size * self.vae_scale_factor
        #width = width or 768 #self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        diffusers.StableDiffusionInpaintPipeline.check_inputs(self,prompt, self.height, self.width, strength, callback_steps) #self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = "cpu" #self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        #text_embeddings = self._encode_prompt(
        #    prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        #)
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        img_buffer = []
        
        text_input = self.tokenizer(prompt)
        text_embeddings = self.text_encoder(text_input)[self._text_encoder_output]
        
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
                
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(uncond_tokens) 
            uncond_embeddings = self.text_encoder(uncond_input)[self._text_encoder_output]
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # 4. Prepare the depth image
        depth = self.prepare_depth(depth_image, image, torch.float, device)

        if image is not None and isinstance(image, PIL.Image.Image):
            image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess(image)

        # 5. set timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        if image is not None:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength,scheduler)

        # 6. Prepare latent variables
        num_channels_latents = 4 #self.vae.config.latent_channels
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        
        latents = self.prepare_latents(image, scheduler, latent_timestep)
        

        # 7. Prepare mask latent variables
        depth = self.prepare_depth_latents(
            depth,
            batch_size * num_images_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        num_channels_depth = depth.shape[1]
        if num_channels_latents + num_channels_depth != 5: #self.unet.config.in_channels:
            raise ValueError(
                f"Select a depth model, such as 'stabilityai/stable-diffusion-2-depth'"
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta,scheduler)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents #torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, depth], dim=1)

                # predict the noise residual
                noise_pred = self.unet([latent_model_input, t, text_embeddings])[self._unet_output]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])
                    

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs).prev_sample.numpy()

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
                # NOTE: Modified to yield the latents instead of calling a callback.
                yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], self.width, self.height, latents, generator, i)

        #self.vae.config.scaling_factor = 8
        # 11. Post-processing
        
        #latents = 1 / self.scaling_factor * latents
        print("-------------before vae_decoder----------------")
        image = self.vae_decoder(latents)[self._vae_d_output]
        print("-------------After vae_decoder-----------------")
        image = np.clip(image / 2 + 0.5, 0, 1) #(image / 2 + 0.5).clamp(0, 1)
        image = np.transpose(image, (0, 2, 3, 1)) #image.cpu().permute(0, 2, 3, 1).float().numpy()
      
        filePath = 'test1_blender.png'
        if os.path.exists(filePath):
            os.remove(filePath)
        
        pil_image = self.numpy_to_pil(image)
        pil_image[0].save(filePath)
        #image = self.decode_latents(latents)

        # TODO: Add UI to enable this.
        # 12. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # NOTE: Modified to yield the decoded image as a numpy array.
        yield ImageGenerationResult(
            [np.asarray(PIL.ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                for i, image in enumerate(self.numpy_to_pil(image))],
            [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
            num_inference_steps,
            True
        )
    


ov_pipe_depth = None

def load_models_depth(self,model_path,infer_device):
    global ov_pipe_depth
    try:
        ov_pipe_depth = StableDiffusionEngineDepth(
            model = model_path,
            device = infer_device)
    except:
        return False
    
    #print("PROCESS ID in print function", os.getpid())
    return True

def depth_to_image(
    self,
    pipeline: Pipeline,
    
    infer_model: model_name_enum,

    scheduler: Scheduler,

    optimizations: Optimizations,

    depth: NDArray | None,
    image: NDArray | str | None,
    strength: float,
    prompt: str | list[str],
    steps: int,
    seed: int,


    infer_model_size: model_size_enum,
    infer_device: device_name_enum,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,

  

    step_preview_mode: StepPreviewMode,

    **kwargs
) -> Generator[NDArray, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:



            
            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()
           

            prediction_type_ = 'epsilon'
            if scheduler == Scheduler.LMS_DISCRETE:
                select_scheduler = LMSDiscreteScheduler
     
            elif scheduler == Scheduler.EULER_DISCRETE:
                select_scheduler = EulerDiscreteScheduler
 
            elif scheduler == Scheduler.DPM_SOLVER_MULTISTEP:
                select_scheduler = DPMSolverMultistepScheduler 
                prediction_type_ = 'v_prediction'
            else:
                select_scheduler = EulerDiscreteScheduler                



            sch = select_scheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear")
       
  

            # RNG
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            generator = []
            for _ in range(batch_size):
                gen = torch.Generator(device="cpu" if device in ("mps", "privateuseone") else device) # MPS and DML do not support the `Generator` API
                generator.append(gen.manual_seed(random.randrange(0, np.iinfo(np.uint32).max) if seed is None else seed))
            if batch_size == 1:
                # Some schedulers don't handle a list of generators: https://github.com/huggingface/diffusers/issues/1909
                generator = generator[0]

            # Init Image
            # FIXME: The `unet.config.sample_size` of the depth model is `32`, not `64`. For now, this will be hardcoded to `512`.
            if infer_model_size.name == "model_size_512":
                height = 512
                width = 512
            else:
                height = 768
                width = 768

 
            rounded_size = (
                int(8 * (width // 8)),
                int(8 * (height // 8)),
            )
            depth_image = PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth * 255)).convert('L')).resize(rounded_size) if depth is not None else None
            init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB').resize(rounded_size)
            if depth_image is None:
                print("DEPTH IS NONE")
            else:
                print("DEPTH IS NOT NONE..its pre generated")
  
            # Inference
            #with (torch.inference_mode() if device not in ('mps', "privateuseone") else nullcontext()), \
                #(torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
            yield from ov_pipe_depth(
                prompt=prompt,
                depth_image=depth_image,
                image=init_image,
                strength=strength,
                scheduler = sch,               
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                negative_prompt=negative_prompt if use_negative_prompt else None,
                num_images_per_prompt=1,
                eta=0.0,
                generator=generator,
                latents=None,
                output_type="pil",
                return_dict=True,
                callback=None,
                callback_steps=1,
                step_preview_mode=step_preview_mode
            )
           

        case Pipeline.STABILITY_SDK:
            import stability_sdk
            raise NotImplementedError()
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")