from typing import Union, Generator, Callable, List, Optional, Dict
import os
from contextlib import nullcontext
import inspect

from numpy.typing import NDArray
import numpy as np
import random
from typing import Union, List, Optional, Tuple
from .prompt_to_image import Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult
from ..models import Pipeline

import inspect
from transformers import CLIPTokenizer

import PIL
import site
from ...absolute_path import absolute_path

site.addsitedir(absolute_path(".python_dependencies"))
#import cv2
from ..models import Pipeline
pipeline = Pipeline.STABLE_DIFFUSION

import diffusers


from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from openvino.runtime import Core
import torch
import time

from PIL import Image, ImageOps


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

def preprocess(image: PIL.Image.Image, ht, wt):
    """
    Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
    then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
    converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
    The function returns preprocessed input tensor and padding size, which can be used in postprocessing.
    
    Parameters:
      image (PIL.Image.Image): input image
    Returns:
       image (np.ndarray): preprocessed image tensor
       meta (Dict): dictionary with preprocessing metadata info
    """
    print("FIRST image size", image.size )
    src_width, src_height = image.size
    #image = image.convert('RGB')
    #image = Image.fromarray(image).convert('RGB')
    #src_width, src_height = image.size
    dst_width, dst_height = scale_fit_to_window(
        wt, ht, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height),
                     resample=PIL.Image.Resampling.LANCZOS))[None, :]

    pad_width = wt - dst_width
    pad_height = ht - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image = image.transpose(0, 3, 1, 2)

    return image, {"padding": pad, "src_width": src_width, "src_height": src_height}

class StableDiffusionEngine(diffusers.DiffusionPipeline):
    @torch.no_grad()
    def __init__(
        self,
     
        model="bes-dev/stable-diffusion-v1-4-openvino",
        
        device=["CPU","CPU","CPU","CPU"]
       
    ):  
        try: 
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.tokenizer.save_pretrained(model)
   

        #print("PROCESS ID in Engine", os.getpid())
        print("Starting Model load")
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        self.text_encoder = self.core.compile_model(os.path.join(model, "text_encoder.xml"), device[0])
        self._text_encoder_output = self.text_encoder.output(0)


        print("unet Device:",device[1])
        print("unet-neg Device:",device[2])

        start = time.time()
        self.unet_time_proj = self.core.compile_model(os.path.join(model, "unet_time_proj.xml"), 'CPU')
        print("compile_model for unet_time_proj on 'CPU'  ", time.time() - start)        

     
        start = time.time()
        self.vae_decoder = self.core.compile_model(os.path.join(model, "vae_decoder.xml"), device[3])
        print("compile_model vae decoder ", time.time() - start)  
        start = time.time()
        self.vae_encoder = self.core.compile_model(os.path.join(model, "vae_encoder.xml"), device[3]) 
        print("compile_model vae decoder ", time.time() - start) 
        print("vae compiled")
        blob_name = "unet_int8_NPU.blob"
        #try: 
        #    print("VERSION OV",self.core.get_versions("NPU"))
        #except Exception as error: 
        #    print("An exception occurred:", error)
        
       
        print(" ALL DEVICES", self.core.available_devices )
        #with open(os.path.join(model, blob_name), "rb") as f:
        #    print("In open model blob" )
        #    try: 

                #self.unet_npu = self.core.import_model(f.read(), "NPU")
           # except Exception as error: 
           #     print("An exception occurred:", error)            
           # print("After import open model blob" ) 
        #print("was it success")      
        
       

        unet_int8_model = "unet_int8.xml"

        if device[1] == "NPU" or device[2] == "NPU":
            device_npu = "NPU"
            blob_name = "unet_int8_NPU.blob"
            print("Loading unet blob on npu:",os.path.join(model, blob_name))
            start = time.time()
            with open(os.path.join(model, blob_name), "rb") as f:
                print("In open model blob" )
                try:
                    self.unet_npu = self.core.import_model(f.read(), "NPU")
                except Exception as error:
                    print("An exception occurred:", error)
                print("After import open model blob success")
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


        # Positive prompt
        if device[1] == "NPU":
            self.unet = self.unet_npu
        elif device[1] == "GPU":
            self.unet = self.unet_gpu
        else:
            self.unet = self.unet_cpu

        # Negative prompt:
        if device[2] == "NPU":
            self.unet_neg = self.unet_npu
        elif device[2] == "GPU":
            self.unet_neg = self.unet_gpu
        else:
            self.unet_neg = self.unet_cpu                        

        self.init_image_shape = tuple(self.vae_encoder.inputs[0].shape)[2:]

        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None

        if self.unet.input("latent_model_input").shape[1] == 4:
            self.height = self.unet.input("latent_model_input").shape[2] * 8
            self.width = self.unet.input("latent_model_input").shape[3] * 8
        else:

            self.height = self.unet.input("latent_model_input").shape[1] * 8
            self.width = self.unet.input("latent_model_input").shape[2] * 8

        self.infer_request_neg = self.unet_neg.create_infer_request()
        self.infer_request = self.unet.create_infer_request()
        self.infer_request_time_proj = self.unet_time_proj.create_infer_request()
        self.time_proj_constants = np.load(os.path.join(model, "time_proj_constants.npy"))               



    
        super().__init__()
    
        self.vae_scale_factor = 8
        self.scaling_factor =  0.18215 

    



    def __call__(
        self,
        prompt: Union[str, List[str]],
        scheduler=None,
        strength = 0.5,
        init_image = Union[torch.FloatTensor, PIL.Image.Image],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        seed=None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # 0. Default height and width to unet
        #height = height or 768  #self.unet.config.sample_size * self.vae_scale_factor
        #width = width or 768 #self.unet.config.sample_size * self.vae_scale_factor

        if init_image is None:
            strength = 1.0
    

        # 1. Check inputs. Raise error if not correct 
        StableDiffusionPipeline.check_inputs(self,prompt, self.height, self.width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = "cpu" #self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # text_embeddings = self._encode_prompt(
        #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        # )
        if seed is not None:
            np.random.seed(seed)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[self._text_encoder_output]
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
                
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
                
            tokens_uncond = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length, #truncation=True,  
                return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(tokens_uncond.input_ids)[self._text_encoder_output]
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # 4. Prepare timesteps
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}

        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        #strength = 1.0
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        latent_timestep = timesteps[:1]

        #timesteps = scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = 4 
        latents, meta = self.prepare_latents(init_image, latent_timestep, scheduler)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = {} 

        # 7. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            noise_pred = []
            latent_model_input = latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            latent_model_input_neg = latent_model_input
            if self.unet.input("latent_model_input").shape[1] != 4:
                #print("In transpose")
                try:
                    latent_model_input = latent_model_input.permute(0,2,3,1)
                except:
                    latent_model_input = latent_model_input.transpose(0,2,3,1)

            if self.unet_neg.input("latent_model_input").shape[1] != 4:
                #print("In transpose")
                try:
                    latent_model_input_neg = latent_model_input_neg.permute(0,2,3,1)
                except:
                    latent_model_input_neg = latent_model_input_neg.transpose(0,2,3,1)



     
            t_scaled = self.time_proj_constants * np.float32(t)

            cosine_t = np.cos(t_scaled)
            sine_t = np.sin(t_scaled)

            time_proj_dict = {"sine_t" : np.float32(sine_t), "cosine_t" : np.float32(cosine_t)}
            self.infer_request_time_proj.start_async(time_proj_dict)
            self.infer_request_time_proj.wait()
            time_proj = self.infer_request_time_proj.get_output_tensor(0).data.astype(np.float32)

            swap = True

             #Alternating between -prompt and +prompt on iGPI and NPU
            if swap:

                if i % 2 == 0:

                    input_tens_neg_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input_neg, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0)}
                    input_tens_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0)}

                else:

                    input_tens_neg_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input_neg, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0)}
                    input_tens_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0)}

            else:
                    input_tens_neg_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input_neg, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0)}
                    input_tens_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0)}


            self.infer_request_neg.start_async(input_tens_neg_dict)
            self.infer_request.start_async(input_tens_dict)
            self.infer_request_neg.wait()
            self.infer_request.wait()

            if swap:

                if i % 2 == 0:

                    noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
                    noise_pred_pos = self.infer_request.get_output_tensor(0)
                else:
                    noise_pred_neg = self.infer_request.get_output_tensor(0)
                    noise_pred_pos = self.infer_request_neg.get_output_tensor(0)
            else:
                    noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
                    noise_pred_pos = self.infer_request.get_output_tensor(0)




            noise_pred.append(noise_pred_neg.data.astype(np.float32))
            noise_pred.append(noise_pred_pos.data.astype(np.float32))



            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs).prev_sample.numpy()

            # NOTE: Modified to yield the latents instead of calling a callback.
            if init_image is not None:
                src_width, src_height = init_image.size
                yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], src_width, src_height, latents, generator, i)
            else:
                yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], self.width, self.height, latents, generator, i)

        # 8. Post-processing

        print("-------------before vae_decoder----------------")
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self._vae_d_output]

        image = self.postprocess_image(image, meta)

        #---------------------Debug-----------------
        #filePath = 'p2i_blender.png'
        #if os.path.exists(filePath):
            #os.remove(filePath)
        
        #pil_image = self.numpy_to_pil(image)
        #pil_image[0].save(filePath)
        #image[0].save(filePath)
        #------------------------------------------
        # TODO: Add UI to enable this.
        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # NOTE: Modified to yield the decoded image as a numpy array.
        if init_image is not None:
            yield ImageGenerationResult(
                [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                    for i, image in enumerate(image)], #self.numpy_to_pil(image))],
                [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
                num_inference_steps,
                True
            )
        else:
            yield ImageGenerationResult(
                [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                    for i, image in enumerate(self.numpy_to_pil(image))],
                [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
                num_inference_steps,
                True
            )

  



    def prepare_latents(self, image:PIL.Image.Image = None, latent_timestep:torch.Tensor = None, scheduler = LMSDiscreteScheduler): #, height, width):
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
                if isinstance(scheduler, LMSDiscreteScheduler):
                    noise = noise * scheduler.sigmas[0].numpy()
                    return noise, {}                    
                elif isinstance(scheduler, EulerDiscreteScheduler):
                    noise = noise * scheduler.sigmas.max().numpy()
                    return noise, {}
                else:
                    return noise, {}
            input_image, meta = preprocess(image,self.height,self.width)  
            moments = self.vae_encoder(input_image)[self._vae_e_output]
            mean, logvar = np.split(moments, 2, axis=1)
            std = np.exp(logvar * 0.5)
            latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
            latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
            return latents, meta
    
    def postprocess_image(self, image:np.ndarray, meta:Dict):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required), 
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format
        
        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images

                        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = [cv2.resize(img, (orig_width, orig_height))
                        for img in image]
    
        return image
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            #print("image shape",image.shape[2:])
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
    
           

        
        if "src_height" in meta:
                image = self.numpy_to_pil(image)
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [img.resize((orig_width, orig_height),
                                    PIL.Image.Resampling.LANCZOS) for img in image]
                return image #np.array(image)
        
        return image



    def get_timesteps(self, num_inference_steps:int, strength:float, scheduler):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength
        
        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
   
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start 


ov_pipe = None

def load_models_int8(self,model_path,infer_device):
    global ov_pipe
    try:
        ov_pipe = StableDiffusionEngine(
            model = model_path,
            device = infer_device)
    except:
        print("Model Didnt compile")
        return False

    
    #print("PROCESS ID in print function", os.getpid())
    return True


def prompt_to_image_int8(
    self,
    #ov_pipe,
    pipeline: Pipeline,
    

    image: NDArray,
    scheduler: Scheduler,
    strength: float,
    optimizations: Optimizations,
    prompt: str | list[str],
    steps: int,
    seed: int,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,
    


    step_preview_mode: StepPreviewMode,



    **kwargs
) -> Generator[ImageGenerationResult, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:


            # Mostly copied from `diffusers.StableDiffusionPipeline`, with slight modifications to yield the latents at each step.
            #class GeneratorPipeline(diffusers.StableDiffusionPipeline):
            

            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device() 
  
 

            print("scheduler", scheduler)
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

            if image is not None:
                # Init Image
                init_image = Image.fromarray(image).convert('RGB') #(numpy to pil)
            else:
                init_image = None



            # RNG
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            generator = []
            for _ in range(batch_size):
                gen = torch.Generator(device="cpu" if device in ("mps", "privateuseone") else device) # MPS and DML do not support the `Generator` API
                generator.append(gen.manual_seed(random.randrange(0, np.iinfo(np.uint32).max) if seed is None else seed))
            if batch_size == 1:
                # Some schedulers don't handle a list of generators: https://github.com/huggingface/diffusers/issues/1909
                generator = generator[0]
            
            #print("PROCESS ID in prompt to image", os.getpid())

            global ov_pipe
 
            yield from ov_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                scheduler=sch,
                init_image=init_image,
                guidance_scale=cfg_scale,
                strength=strength,
                negative_prompt=negative_prompt if use_negative_prompt else None,
                num_images_per_prompt=1,
                eta=0.0,
                generator=generator,
                seed=seed,
                latents=None,
                output_type="pil",
                return_dict=True,
                callback=None,
                callback_steps=1,
                step_preview_mode=step_preview_mode
            )

          
        case Pipeline.STABILITY_SDK:
            print("NOT important")
           
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")


