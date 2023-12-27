from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext
import inspect

from numpy.typing import NDArray
import numpy as np
import random
from typing import Union, List, Optional, Tuple
from .prompt_to_image import Scheduler, model_size_enum, Optimizations, StepPreviewMode, ImageGenerationResult
from ..models import Pipeline


#from diffusers import StableDiffusionDepth2ImgPipeline
# tokenizer

from pathlib import Path

pipeline = Pipeline.STABLE_DIFFUSION

import diffusers

from diffusers import UniPCMultistepScheduler
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from openvino.runtime import Core, Model
from transformers import CLIPTokenizer
import torch
import time

import PIL.Image
import PIL.ImageOps
 
def scale_fit_to_window(dst_width:int, dst_height:int, image_width:int, image_height:int):
    """
    Preprocessing helper function for calculating image size for resize with peserving original aspect ratio 
    and fitting image to specific window size
    
    Parameters:
      dst_width (int): destination window width
      dst_height (int): destination window height
      image_width (int): source image width
      image_height (int): source image height
    Returns:
      result_width (int): calculated width for resize
      result_height (int): calculated height for resize
    """
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)

def randn_tensor(
    shape: Union[Tuple, List],
    dtype: Optional[np.dtype] = np.float32,
):
    """
    Helper function for generation random values tensor with given shape and data type
    
    Parameters:
      shape (Union[Tuple, List]): shape for filling random values
      dtype (np.dtype, *optiona*, np.float32): data type for result
    Returns:
      latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
    """
    latents = np.random.randn(*shape).astype(dtype)

    return latents


def preprocess(image: PIL.Image.Image):
    """
    Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
    then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
    converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
    The function returns preprocessed input tensor and padding size, which can be used in postprocessing.
    
    Parameters:
      image (Image.Image): input image
    Returns:
       image (np.ndarray): preprocessed image tensor
       pad (Tuple[int]): pading size for each dimension for restoring image size in postprocessing
    """
    src_width, src_height = image.size
    dst_width, dst_height = scale_fit_to_window(512, 512, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height), resample=PIL.Image.Resampling.LANCZOS))[None, :]
    pad_width = 512 - dst_width
    pad_height = 512 - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = image.transpose(0, 3, 1, 2)
    return image, pad

class StableDiffusionEngineControlDepth(diffusers.DiffusionPipeline):
    def __init__(
        self,
        model="bes-dev/stable-diffusion-v1-4-openvino",
        device=["CPU","CPU","CPU","CPU"]
     
    ):   
        
        super().__init__()

        tokenizer = "openai/clip-vit-large-patch14"

        self.vae_scale_factor = 8        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)
            
        print("Starting Controlnet Depth Model load int8")

        self.core = Core()
       
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        print("Setting Cache done")

        print("MODEL Path",model)
        controlnet = os.path.join(model, "controlnet-depth.xml")
        text_encoder = os.path.join(model, "text_encoder.xml")
        unet_int8_model = os.path.join(model, "unet_controlnet_int8.xml")
        unet_time_proj_model = os.path.join(model, "unet_time_proj_sym.xml")
        vae_decoder = os.path.join(model, "vae_decoder.xml")



        ############################################################
        self.load_models(self.core, device, controlnet, text_encoder, unet_time_proj_model, unet_int8_model, vae_decoder, model)

        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8 
        
       

        self.infer_request_neg = self.unet_neg.create_infer_request()
        self.infer_request = self.unet.create_infer_request()
        self.infer_request_time_proj = self.unet_time_proj.create_infer_request()
        self.infer_request_controlnet = self.controlnet.create_infer_request()
   
        
        print("All models loaded")


    def load_models(self, core: Core, device: str, controlnet:Model, text_encoder: Model, unet_time_proj_model:Model, unet_int8_model: Model, vae_decoder: Model, model: str):
        start = time.time()
        self.text_encoder = self.core.compile_model(text_encoder, device[0])
        print("compiling done for text_encoder in", time.time() - start)

        self._text_encoder_output = self.text_encoder.output(0)

        start = time.time()
        self.controlnet = self.core.compile_model(controlnet, "GPU")  
        print("compiling done for controlnet in", time.time() - start)      


        start = time.time()
        self.vae_decoder = self.core.compile_model(vae_decoder, device[3])
        print("compiling done for vae_decoder in", time.time() - start)
        
        self._vae_d_output = self.vae_decoder.output(0)
  
        self.unet_time_proj = self.core.compile_model(unet_time_proj_model, "CPU")  




        if device[1] == "NPU" or device[2] == "NPU":
            device_npu = "NPU"
            blob_name = "unet_controlnet_int8_NPU.blob"
            print("Loading unet blob on npu:",blob_name)
            start = time.time()
            with open(os.path.join(model, blob_name), "rb") as f:
                self.unet_npu = self.core.import_model(f.read(), device_npu)
            print("unet loaded on npu in:", time.time() - start)
            
        if device[1] == "GPU" or device[2] == "GPU":
            print("compiling start on GPU")
            start = time.time()
            self.unet_gpu = self.core.compile_model(os.path.join(model, unet_int8_model), "GPU")
            print("compiling done on GPU in", time.time() - start)
            
        if device[1] == "CPU" or device[2] == "CPU":
            print("compiling start on CPU")
            start = time.time()
            self.unet_cpu = self.core.compile_model(os.path.join(model, unet_int8_model), "CPU")
            print("compiling done on CPU in", time.time() - start)
        print("Controlnet Depth Model Compiled")

       
        # Positive prompt
        if device[1] == "NPU":
       
            self.unet = self.unet_npu
            self.npu_flag = True
         
        elif device[1] == "GPU":
            self.unet = self.unet_gpu
            self.npu_flag = False

        else:
            self.unet = self.unet_cpu
            self.npu_flag = False

        # Negative prompt:
        if device[2] == "NPU":
            self.unet_neg = self.unet_npu
           
            self.npu_flag_neg = True
  
        elif device[2] == "GPU":
            self.unet_neg = self.unet_gpu
            self.npu_flag_neg = False
        else:
            self.unet_neg = self.unet_cpu
            self.npu_flag_neg = False

       
    def prepare_depth(self, depth, image, dtype, device):
        device = torch.device('cpu') # if device.type == 'mps' else device.type)
       
        if depth is None:
            print("Depth is None so Depth setting up the depth model")
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            import contextlib
      
            
            feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas") 
            depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas") 
            depth_estimator = depth_estimator.to(device)

            if isinstance(image, PIL.Image.Image):
                image = [image]
            else:
                image = [img for img in image]

            if isinstance(image[0], PIL.Image.Image):
                width, height = image[0].size
            else:
                width, height = image[0].shape[-2:]            
            
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
         
            pixel_values = pixel_values.to(device=device)
            # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
            # So we use `torch.autocast` here for half precision inference.
            #context_manger = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
         
            depth_map = depth_estimator(pixel_values).predicted_depth
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=image[0].size[::-1], #(self.height // self.vae_scale_factor, self.width // self.vae_scale_factor),
                mode="bicubic",
                align_corners=False,
            )



            output = depth_map.squeeze().cpu().detach().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_output = PIL.Image.fromarray(formatted)
            
            image_d = np.array(depth_output)
            image_d = image_d[:, :, None]
            image_d = np.concatenate([image_d, image_d, image_d], axis=2)
            depth_image = PIL.Image.fromarray(image_d)            
            return depth_image
        else:
            if isinstance(depth, PIL.Image.Image):
                image_d = np.array(depth)
                image_d = image_d[:, :, None]
                image_d = np.concatenate([image_d, image_d, image_d], axis=2)
                depth_image = PIL.Image.fromarray(image_d)



                return depth_image
            



   
    def prepare_latents(self, height, width,scheduler):
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
        shape = (1, 4, height // self.vae_scale_factor, width // self.vae_scale_factor)
    
        latents = randn_tensor(shape, np.float32)
    

        # scale the initial noise by the standard deviation required by the scheduler
        if isinstance(scheduler, LMSDiscreteScheduler):
            
            latents = latents * scheduler.sigmas[0].numpy()
        elif isinstance(scheduler, EulerDiscreteScheduler):
            
            latents = latents * scheduler.sigmas.max().numpy()
        else:
            latents = latents * scheduler.init_noise_sigma
        return latents
    

    def decode_latents(self, latents:np.array, pad:Tuple[int]):
        """
        Decode predicted image from latent space using VAE Decoder and unpad image result
        
        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
           pad (Tuple[int]): each side padding sizes obtained on preprocessing step
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self._vae_d_output]
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image    

    def _encode_prompt(self, prompt:Union[str, List[str]], num_images_per_prompt:int = 1, do_classifier_free_guidance:bool = True, negative_prompt:Union[str, List[str]] = None):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids)[self._text_encoder_output]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(
                text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(
                text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings       
    
    
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
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)              
    
        #image = image.convert("RGB")
        #depth_image = prepare_depth(image)

        # 4. Prepare the depth image
        
        depth = self.prepare_depth(depth_image, image, torch.float, device)

        orig_width, orig_height = depth.size
        depth, pad = preprocess(depth)
        height, width = depth.shape[-2:]

        if depth_image is None:
            latent_ht = height
            latent_wt = width
        else:
            latent_ht = self.height
            latent_wt = self.width
    
        if do_classifier_free_guidance:
            depth = np.concatenate(([depth] * 2))        



        # 5. set timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
 

        # 6. Prepare latent variables
        num_channels_latents = 4 #self.vae.config.latent_channels

        
        latents = self.prepare_latents(latent_ht,latent_wt,scheduler)
        
        controlnet_conditioning_scale = 1.0

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
       # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta,scheduler)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):


                noise_pred = []
                latent_model_input = latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                latent_model_input_2 = np.concatenate(
                    [latents] * 2) if do_classifier_free_guidance else latents    
                    
                latent_model_input_2 = scheduler.scale_model_input(latent_model_input_2, t)


       
                #result = self.controlnet([latent_model_input_2, t, text_embeddings, pose])  
                controlnet_dict = {"sample":latent_model_input_2, "timestep":t, "encoder_hidden_states":text_embeddings, "controlnet_cond":depth}
                result = self.infer_request_controlnet.infer(controlnet_dict, share_outputs = True)

                tensor_dict_neg = {}
                tensor_dict = {}

                for k,v in result.items():
                    tensor_name = next(iter(k.names))
                    #print("tensor_name--", tensor_name)
                    vneg = np.expand_dims(v[0], axis=0)
                    tensor_dict_neg[tensor_name] = vneg #controlnet_conditioning_scale * vneg #.astype(np.float32)
                    
                    vpos = np.expand_dims(v[1], axis=0)
                    tensor_dict[tensor_name] = vpos #controlnet_conditioning_scale * vpos #.astype(np.float32)  

                time_proj_dict = {"timestep" : t}
                self.infer_request_time_proj.start_async(time_proj_dict,share_inputs = True)
                self.infer_request_time_proj.wait()
                time_proj = self.infer_request_time_proj.get_output_tensor(0).data.astype(np.float32) 
                
                ##### NEGATIVE PIPELINE #####
                input_dict_neg = {"sample":latent_model_input, "time_proj": time_proj, "encoder_hidden_states":np.expand_dims(text_embeddings[0], axis=0)}
                input_dict_neg.update(tensor_dict_neg)
                
                if self.npu_flag_neg:
                    input_dict_neg_final = {k: v for k, v in sorted(input_dict_neg.items(), key=lambda x: x[0])}
                else:
                    input_dict_neg_final = input_dict_neg
                
                self.infer_request_neg.start_async(input_dict_neg_final, share_inputs = True)
                
                
                ##### POSITIVE PIPELINE #####
                input_dict = {"sample":latent_model_input, "time_proj": time_proj, "encoder_hidden_states":np.expand_dims(text_embeddings[1], axis=0)}
                input_dict.update(tensor_dict)
                if self.npu_flag:
                    input_dict_final = {k: v for k, v in sorted(input_dict.items(), key=lambda x: x[0])}
                else:
                    input_dict_final = input_dict

                self.infer_request.start_async(input_dict_final,share_inputs = True)
                self.infer_request_neg.wait()
                self.infer_request.wait()                    
                
                noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
                noise_pred_pos = self.infer_request.get_output_tensor(0) 

                noise_pred.append(noise_pred_neg.data.astype(np.float32))
                noise_pred.append(noise_pred_pos.data.astype(np.float32))  

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)                     

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
                # NOTE: Modified to yield the latents instead of calling a callback.
                yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], self.width, self.height, latents, generator, i)

        #self.vae.config.scaling_factor = 8
        # 11. Post-processing

        image = self.decode_latents(latents, pad)        
        

        filePath = 'test1_blender.png'
        if os.path.exists(filePath):
            os.remove(filePath)
        
        pil_image = self.numpy_to_pil(image)
        pil_image[0].save(filePath)
        #image = self.decode_latents(latents)

        image = self.numpy_to_pil(image)
        #print("DEPTH IMAGEEEEEE",depth_image)
        if depth_image is None:
            image = [img.resize((orig_width, orig_height), PIL.Image.Resampling.LANCZOS) for img in image]        

        # TODO: Add UI to enable this.
        # 12. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # NOTE: Modified to yield the decoded image as a numpy array.
        yield ImageGenerationResult(
            [np.asarray(PIL.ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                for i, image in enumerate(image)], #self.numpy_to_pil(image))],
            [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
            num_inference_steps,
            True
        )
   


ov_pipe_depth = None

def load_models_depth_int8(self,model_path,infer_device):
    global ov_pipe_depth
    try:
        ov_pipe_depth = StableDiffusionEngineControlDepth(
            model = model_path,
            device = infer_device)
    except Exception as error:
        print("Model Didnt compile Error -",error)
        return False
    
    #print("PROCESS ID in print function", os.getpid())
    return True

def depth_to_image_int8(
    self,
    pipeline: Pipeline,
    

    scheduler: Scheduler,

    optimizations: Optimizations,

    depth: NDArray | None,
    image: NDArray | str | None,
    strength: float,
    prompt: str | list[str],
    steps: int,
    seed: int,


    infer_model_size: model_size_enum,
 

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
            elif scheduler == Scheduler.UNI_PC_MULTISTEP:
                select_scheduler = UniPCMultistepScheduler
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
           
            init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB') #None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB').resize(rounded_size)
            if depth_image is None:
                print("DEPTH IS NONE")
            elif init_image is None:
                print("INITIAL IMAGE IS NONE")
            else:
                print("DEPTH IS NOT NONE..its pre generated")
  
   
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