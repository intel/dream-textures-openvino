from typing import Annotated, Union, _AnnotatedAlias, Generator, Callable, List, Optional, Any, Dict
import enum
import math
import os
import sys
from dataclasses import dataclass
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random


import site
import inspect




from ...absolute_path import absolute_path
site.addsitedir(absolute_path(".python_dependencies"))
#import cv2
import PIL
from PIL import Image, ImageOps
from ..models import Pipeline


pipeline = Pipeline.STABLE_DIFFUSION

#match pipeline:
#    case Pipeline.STABLE_DIFFUSION:
import diffusers
from diffusers import StableDiffusionPipeline
#from openvino.runtime import Model, Core
from transformers import CLIPTokenizer
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
import torch

        
print ("**********************************************CALL 1 *******************************")

from openvino.runtime import Core
ie = Core()
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    print(user_paths)
except KeyError:
    user_paths = []

try: 
    print("VERSION OV", ie.get_versions("NPU"))
except Exception as error: 
    print("An exception occurred:", error)
            




class CachedPipeline:
    """A pipeline that has been cached for subsequent runs."""

    pipeline: Any
    """The diffusers pipeline to re-use"""

    invalidation_properties: tuple
    """Values that, when changed, will invalid this cached pipeline"""

    snapshot_folder: str
    """The snapshot folder containing the model"""

    def __init__(self, pipeline: Any, invalidation_properties: tuple, snapshot_folder: str):
        self.pipeline = pipeline
        self.invalidation_properties = invalidation_properties
        self.snapshot_folder = snapshot_folder

    def is_valid(self, properties: tuple):
        return properties == self.invalidation_properties

def load_pipe(self, action, generator_pipeline, model, optimizations, scheduler, device):
    """
    Use a cached pipeline, or create the pipeline class and cache it.
    
    The cached pipeline will be invalidated if the model or use_cpu_offload options change.
    """
    import torch
    import gc

    invalidation_properties = (
        action, model, device,
        optimizations.can_use("sequential_cpu_offload", device),
        optimizations.can_use("half_precision", device),
    )
    print("----In load pipe-----")
    cached_pipe: CachedPipeline = self._cached_pipe if hasattr(self, "_cached_pipe") else None
    if cached_pipe is not None and cached_pipe.is_valid(invalidation_properties):
        pipe = cached_pipe.pipeline
    else:
        # Release the cached pipe before loading the new one.
        if cached_pipe is not None:
            del self._cached_pipe
            del cached_pipe
            gc.collect()

        revision = "fp16" if optimizations.can_use_half(device) else None
        snapshot_folder = model_snapshot_folder(model, revision)
        pipe = generator_pipeline.from_pretrained(
            snapshot_folder,
            revision=revision,
            torch_dtype=torch.float16 if optimizations.can_use_half(device) else torch.float32,
        )
        pipe = pipe.to(device)
        setattr(self, "_cached_pipe", CachedPipeline(pipe, invalidation_properties, snapshot_folder))
        cached_pipe = self._cached_pipe
    if 'scheduler' in os.listdir(cached_pipe.snapshot_folder):
        pipe.scheduler = scheduler.create(pipe, {
            'model_path': cached_pipe.snapshot_folder,
            'subfolder': 'scheduler',
        })
    else:
        pipe.scheduler = scheduler.create(pipe, None)
    return pipe

#ie = Core()
supported_devices = ie.available_devices

device_dict = dict()
device_dict_npu = dict()
for d in supported_devices:   
    if d == "GPU" or d == "NPU":
        device_dict_npu[d] = d
    if d != "NPU" :
        if "GNA" not in d:
            device_dict[d] = d


#print("device_dict",device_dict)
#print("device_dict_npu", device_dict_npu)

device_name_enum = enum.Enum('device_name_enum', device_dict)
device_name_enum_npu = enum.Enum('device_name_enum_npu', device_dict_npu)


class model_name_enum(enum.Enum):
    Stable_Diffusion_1_5 = "Stable_Diffusion_1_5"
    Stable_Diffusion_1_5_int8 = "Stable_Diffusion_1_5_int8"
    Stable_Diffusion_1_5_controlnet_depth_int8 = "Stable_Diffusion_1_5_controlnet_depth_int8"
    Stable_Diffusion_1_5_controlnet_depth = "Stable_Diffusion_1_5_controlnet_depth"


class model_size_enum(enum.Enum):
    model_size_512 = "512x512"
    #model_size_768 = "768x768"




class Scheduler(enum.Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    DEIS_MULTISTEP = "DEIS Multistep"
    DPM_SOLVER_MULTISTEP = "DPM Solver Multistep"
    DPM_SOLVER_SINGLESTEP = "DPM Solver Singlestep"
    EULER_DISCRETE = "Euler Discrete"
    EULER_ANCESTRAL_DISCRETE = "Euler Ancestral Discrete"
    HEUN_DISCRETE = "Heun Discrete"
    KDPM2_DISCRETE = "KDPM2 Discrete" # Non-functional on mps
    KDPM2_ANCESTRAL_DISCRETE = "KDPM2 Ancestral Discrete"
    LMS_DISCRETE = "LMS Discrete"
    PNDM = "PNDM"
    UNI_PC_MULTISTEP = "UniPCMultistepScheduler"

    def create(self, pipeline, pretrained):
        import diffusers
        def scheduler_class():
            match self:
                case Scheduler.DDIM:
                    return diffusers.schedulers.DDIMScheduler
                case Scheduler.DDPM:
                    return diffusers.schedulers.DDPMScheduler
                case Scheduler.DEIS_MULTISTEP:
                    return diffusers.schedulers.DEISMultistepScheduler
                case Scheduler.DPM_SOLVER_MULTISTEP:
                    return diffusers.schedulers.DPMSolverMultistepScheduler
                case Scheduler.DPM_SOLVER_SINGLESTEP:
                    return diffusers.schedulers.DPMSolverSinglestepScheduler
                case Scheduler.EULER_DISCRETE:
                    return diffusers.schedulers.EulerDiscreteScheduler
                case Scheduler.EULER_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.EulerAncestralDiscreteScheduler
                case Scheduler.HEUN_DISCRETE:
                    return diffusers.schedulers.HeunDiscreteScheduler
                case Scheduler.KDPM2_DISCRETE:
                    return diffusers.schedulers.KDPM2DiscreteScheduler
                case Scheduler.KDPM2_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.KDPM2AncestralDiscreteScheduler
                case Scheduler.LMS_DISCRETE:
                    return diffusers.schedulers.LMSDiscreteScheduler
                case Scheduler.PNDM:
                    return diffusers.schedulers.PNDMScheduler
                case Scheduler.UNI_PC_MULTISTEP:
                    return diffusers.schedulers.UniPCMultistepScheduler
                                
        if pretrained is not None:
            print("pretrained is not None")
            return scheduler_class().from_pretrained(pretrained['model_path'], subfolder=pretrained['subfolder'])
        else:
            print("pretrained is None",pipeline.scheduler.config)
            return scheduler_class().from_config(pipeline.scheduler.config)
    
    def stability_sdk(self):
        import stability_sdk.interfaces.gooseai.generation.generation_pb2
        match self:
            case Scheduler.LMS_DISCRETE:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_K_LMS
            case Scheduler.DDIM:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_DDIM
            case Scheduler.DDPM:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_DDPM
            case Scheduler.EULER_DISCRETE:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_K_EULER
            case Scheduler.EULER_ANCESTRAL_DISCRETE:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_K_EULER_ANCESTRAL
            case _:
                raise ValueError(f"{self} cannot be used with DreamStudio.")

@dataclass(eq=True)
class Optimizations:
    attention_slicing: bool = True
    attention_slice_size: Union[str, int] = "auto"
    cudnn_benchmark: Annotated[bool, "cuda"] = False
    tf32: Annotated[bool, "cuda"] = False
    amp: Annotated[bool, "cuda"] = False
    half_precision: Annotated[bool, {"cuda", "privateuseone"}] = True
    sequential_cpu_offload: Annotated[bool, {"cuda", "privateuseone"}] = False
    channels_last_memory_format: bool = False
    # xformers_attention: bool = False # FIXME: xFormers is not yet available.
    batch_size: int = 1
    vae_slicing: bool = True

    cpu_only: bool = False

    @staticmethod
    def infer_device() -> str:
        if sys.platform == "darwin":
            return "mps"
        elif Pipeline.directml_available():
            return "privateuseone"
        else:
            return "cuda"

    def can_use(self, property, device) -> bool:
        if not getattr(self, property):
            return False
        if isinstance(self.__annotations__.get(property, None), _AnnotatedAlias):
            annotation: _AnnotatedAlias = self.__annotations__[property]
            opt_dev = annotation.__metadata__[0]
            if isinstance(opt_dev, str):
                return opt_dev == device
            return device in opt_dev
        return True

    def can_use_half(self, device):
        if self.half_precision and device == "cuda":
            import torch
            name = torch.cuda.get_device_name()
            return not ("GTX 1650" in name or "GTX 1660" in name)
        return self.can_use("half_precision", device)
    
    def apply(self, pipeline, device):
        """
        Apply the optimizations to a diffusers pipeline.

        All exceptions are ignored to make this more general purpose across different pipelines.
        """
        import torch

        torch.backends.cudnn.benchmark = self.can_use("cudnn_benchmark", device)
        torch.backends.cuda.matmul.allow_tf32 = self.can_use("tf32", device)

        try:
            if self.can_use("attention_slicing", device):
                pipeline.enable_attention_slicing(self.attention_slice_size)
            else:
                pipeline.disable_attention_slicing()
        except: pass
        
        try:
            if self.can_use("sequential_cpu_offload", device) and device in ["cuda", "privateuseone"]:
                # Doesn't allow for selecting execution device
                # pipeline.enable_sequential_cpu_offload()

                from accelerate import cpu_offload

                for cpu_offloaded_model in [pipeline.unet, pipeline.text_encoder, pipeline.vae]:
                    if cpu_offloaded_model is not None:
                        cpu_offload(cpu_offloaded_model, device)

                if pipeline.safety_checker is not None:
                    cpu_offload(pipeline.safety_checker.vision_model, device)
        except: pass
        
        try:
            if self.can_use("channels_last_memory_format", device):
                pipeline.unet.to(memory_format=torch.channels_last)
            else:
                pipeline.unet.to(memory_format=torch.contiguous_format)
        except: pass

        # FIXME: xFormers wheels are not yet available (https://github.com/facebookresearch/xformers/issues/533)
        # if self.can_use("xformers_attention", device):
        #     pipeline.enable_xformers_memory_efficient_attention()

        try:
            if self.can_use("vae_slicing", device):
                # Not many pipelines implement the enable_vae_slicing()/disable_vae_slicing()
                # methods but all they do is forward their call to the vae anyway.
                pipeline.vae.enable_slicing()
            else:
                pipeline.vae.disable_slicing()
        except: pass
        
        from .. import directml_patches
        if device == "privateuseone":
            directml_patches.enable(pipeline)
        else:
            directml_patches.disable(pipeline)

        return pipeline

class StepPreviewMode(enum.Enum):
    NONE = "None"
    FAST = "Fast"
    #FAST_BATCH = "Fast (Batch Tiled)"
    #ACCURATE = "Accurate"
    #ACCURATE_BATCH = "Accurate (Batch Tiled)"

@dataclass
class ImageGenerationResult:
    images: List[NDArray]
    seeds: List[int]
    step: int
    final: bool

    @staticmethod
    def step_preview(pipe, mode, width, height, latents, generator, iteration):
        from PIL import Image, ImageOps
        seeds = [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()]
        match mode:
            case StepPreviewMode.FAST:
                return ImageGenerationResult(
                    [np.asarray(ImageOps.flip(Image.fromarray(approximate_decoded_latents(latents[-1:]))).resize((width, height), Image.Resampling.NEAREST).convert('RGBA'), dtype=np.float32) / 255.],
                    seeds[-1:],
                    iteration,
                    False
                )
            case StepPreviewMode.FAST_BATCH:
                return ImageGenerationResult(
                    [
                        np.asarray(ImageOps.flip(Image.fromarray(approximate_decoded_latents(latents[i:i + 1]))).resize((width, height), Image.Resampling.NEAREST).convert('RGBA'),
                                   dtype=np.float32) / 255.
                        for i in range(latents.size(0))
                    ],
                    seeds,
                    iteration,
                    False
                )
            case StepPreviewMode.ACCURATE:
                return ImageGenerationResult(
                    [np.asarray(ImageOps.flip(pipe.numpy_to_pil(pipe.decode_latents(latents[-1:]))[0]).convert('RGBA'),
                                dtype=np.float32) / 255.],
                    seeds[-1:],
                    iteration,
                    False
                )
            case StepPreviewMode.ACCURATE_BATCH:
                return ImageGenerationResult(
                    [
                        np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                        for image in pipe.numpy_to_pil(pipe.decode_latents(latents))
                    ],
                    seeds,
                    iteration,
                    False
                )
        return ImageGenerationResult(
            [],
            [seeds],
            iteration,
            False
        )

    def tile_images(self):
        images = self.images
        if len(images) == 0:
            return None
        elif len(images) == 1:
            return images[0]
        width = images[0].shape[1]
        height = images[0].shape[0]
        tiles_x = math.ceil(math.sqrt(len(images)))
        tiles_y = math.ceil(len(images) / tiles_x)
        tiles = np.zeros((height * tiles_y, width * tiles_x, 4), dtype=np.float32)
        bottom_offset = (tiles_x*tiles_y-len(images)) * width // 2
        for i, image in enumerate(images):
            x = i % tiles_x
            y = tiles_y - 1 - int((i - x) / tiles_x)
            x *= width
            y *= height
            if y == 0:
                x += bottom_offset
            tiles[y: y + height, x: x + width] = image
        return tiles

def choose_device(self) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    if Pipeline.directml_available():
        import torch_directml
        if torch_directml.is_available():
            # can be named better when torch.utils.rename_privateuse1_backend() is released
            return "privateuseone"
    return "cpu"

def approximate_decoded_latents(latents):
    """
    Approximate the decoded latents without using the VAE.
    """
    import torch
    # origingally adapted from code by @erucipe and @keturn here:
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7

    # these updated numbers for v1.5 are from @torridgristle
    try:
        v1_5_latent_rgb_factors = torch.tensor([
        #    R        G        B
        [ 0.3444,  0.1385,  0.0670], # L1
        [ 0.1247,  0.4027,  0.1494], # L2
        [-0.3192,  0.2513,  0.2103], # L3
        [-0.1307, -0.1874, -0.7445]  # L4
    ], dtype=torch.float, device="cpu") #latents.device)
        #print("DOUBLE OR FLOAT ----------------------------", latents.dtype)
        latent_image = torch.from_numpy(latents)[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
    except:
        v1_5_latent_rgb_factors = torch.tensor([
        #    R        G        B
        [ 0.3444,  0.1385,  0.0670], # L1
        [ 0.1247,  0.4027,  0.1494], # L2
        [-0.3192,  0.2513,  0.2103], # L3
        [-0.1307, -0.1874, -0.7445]  # L4
    ], dtype=torch.double, device="cpu") #latents.device)
        #print("DOUBLE OR FLOAT in except----------------------------", latents.dtype)
        latent_image = torch.from_numpy(latents)[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
    finally:
        latents_ubyte = (((latent_image + 1) / 2)
                        .clamp(0, 1)  # change scale from -1..1 to 0..1
                        .mul(0xFF)  # to 0..255
                        .byte()).cpu()

    return latents_ubyte.numpy()

def model_snapshot_folder(model, preferred_revision: str | None = None):
    """ Try to find the preferred revision, but fallback to another revision if necessary. """
    import diffusers
    storage_folder = os.path.join(diffusers.utils.DIFFUSERS_CACHE, model)
    if os.path.exists(os.path.join(storage_folder, 'model_index.json')): # converted model
        snapshot_folder = storage_folder
    else: # hub model
        revisions = {}
        for revision in os.listdir(os.path.join(storage_folder, "refs")):
            ref_path = os.path.join(storage_folder, "refs", revision)
            with open(ref_path) as f:
                commit_hash = f.read()

            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if len(os.listdir(snapshot_folder)) > 1:
                revisions[revision] = snapshot_folder

        if len(revisions) == 0:
            return None
        elif preferred_revision in revisions:
            revision = preferred_revision
        elif preferred_revision in [None, "fp16"] and "main" in revisions:
            revision = "main"
        elif preferred_revision in [None, "main"] and "fp16" in revisions:
            revision = "fp16"
        else:
            revision = next(iter(revisions.keys()))
        snapshot_folder = revisions[revision]

    return snapshot_folder




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
    print("2nd image size", image.size )
    pad_width = wt - dst_width
    pad_height = ht - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image = image.transpose(0, 3, 1, 2)
    print("4th image size", image.shape )
    return image, {"padding": pad, "src_width": src_width, "src_height": src_height}

from openvino.runtime import Core
class StableDiffusionEngine(diffusers.DiffusionPipeline):
    @torch.no_grad()
    def __init__(
        self,
     
        model="bes-dev/stable-diffusion-v1-4-openvino",
        
        device=["CPU","CPU","CPU"]
       
    ):  
        try: 
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.tokenizer.save_pretrained(model)
   

        #print("PROCESS ID in Engine", os.getpid())
        print("Starting Model load")
        self.core = Core()
        print(" ALL DEVICES", ie.available_devices )
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        self.text_encoder = self.core.compile_model(os.path.join(model, "text_encoder.xml"), device[0])
        self._text_encoder_output = self.text_encoder.output(0)

        self.unet = self.core.compile_model(os.path.join(model, "unet.xml"), device[1])
        self._unet_output = self.unet.output(0)
        self.latent_shape = tuple(self.unet.inputs[0].shape)[1:]
        self.vae_decoder = self.core.compile_model(os.path.join(model, "vae_decoder.xml"), device[2])
        self.vae_encoder = self.core.compile_model(os.path.join(model, "vae_encoder.xml"), device[2]) 
        print("After all compile")

        self.init_image_shape = tuple(self.vae_encoder.inputs[0].shape)[2:]

        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None

        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8                     



    
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
            # expand the latents if we are doing classifier free guidance
            #latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet([latent_model_input, t, text_embeddings])[self._unet_output]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            #latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
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
        #if "src_height" in meta:
        #    orig_height, orig_width = meta["src_height"], meta["src_width"]

            ##image = image.resize(orig_width, orig_height)
            #image = cv2.resize(image, (orig_width, orig_height))
                        
        

        
                      #image = (image / 2 + 0.5).clip(0, 1)
        #image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)   


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

def load_models(self,model_path,infer_device):
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


def prompt_to_image(
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
