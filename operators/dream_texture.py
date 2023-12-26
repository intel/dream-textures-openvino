import bpy
import hashlib
import numpy as np
import math
import os
from pathlib import Path


from .notify_result import NotifyResult

from ..pil_to_image import *
from ..prompt_engineering import *
from ..generator_process import Generator

from ..generator_process.actions.prompt_to_image import ImageGenerationResult, Pipeline, Scheduler

from ..generator_process.actions.huggingface_hub import ModelType
from ..absolute_path import absolute_path


def bpy_image(name, width, height, pixels, existing_image):
    if existing_image is not None and (existing_image.size[0] != width or existing_image.size[1] != height):
        bpy.data.images.remove(existing_image)
        existing_image = None
    if existing_image is None:
        image = bpy.data.images.new(name, width=width, height=height)
    else:
        image = existing_image
        image.name = name
    image.pixels.foreach_set(pixels)
    image.pack()
    image.update()
    return image

def python_exec():
  
    try:
        # 2.92 and older
        path = bpy.app.binary_path_python
    except AttributeError:
        # 2.93 and later
        import sys
        path = sys.executable
    return os.path.abspath(path)

#Models_loaded = False
#Models_loaded_depth = False



class LoadModel(bpy.types.Operator):
    bl_idname = "shade.dream_texture_load_models"
    bl_label = "Load Model"
    bl_description = "Load and compile the SD models"
    bl_options = {'REGISTER'}
    Models_loaded_512 = False
    Models_loaded_512_int8 = False
    Models_loaded_depth_512 = False
    Models_loaded_depth_512_int8 = False


    @classmethod
    def poll(cls, context):
        try:
         
            context.scene.dream_textures_prompt.validate(context)
        except:
            return False
        return Generator.shared().can_use()    

    def execute(self, context):
        scene = context.scene

        generated_args = scene.dream_textures_prompt.generate_args()
        weight_path = scene.dream_textures_prompt.weight_path
        infer_model = generated_args['infer_model']
        infer_model_size = generated_args['infer_model_size'].name
  

        #print("PROCESS ID in Load model", os.getpid())

    

        infer_device_text = generated_args['infer_device_text'].name
        

        infer_device_vae = generated_args['infer_device_vae'].name    

        model_path = Path(weight_path) / infer_model.name / infer_model_size
        self.report({'INFO'}, infer_model.name)
        
        #scene.dream_textures_info = "Loading Models..."


        def exception_callback(_, exception):
            scene.dream_textures_info = ""
            scene.dream_textures_progress = 0
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(exception))
            raise exception
        
        #global Models_loaded,Models_loaded_depth
        def done_callback(future):
            print("Stable diffusion OpenVino Model compiled and loaded")          

        
        try :
            Generator.shared_close()
            kill_loadmodels()
        except:
            pass
        gen = Generator.shared()

        #print("PROCESS ID in gen.prompt_to_image in load", os.getpid())
        print("IN INFER MODEL ",model_path)
        
        if infer_model.name == "Stable_Diffusion_1_5_controlnet_depth_int8":  #"Stable_Diffusion_2_1_depth":
                infer_device_unet_pos = generated_args['infer_device_unet_pos'].name
                infer_device_unet_neg = generated_args['infer_device_unet_neg'].name           
                print("IN INFER MODEL FOR CONTROLNET DEPTH INT8",model_path)
                infer_device = [infer_device_text, infer_device_unet_pos, infer_device_unet_neg, infer_device_vae]
           
                f  = gen.load_models_depth_int8(model_path,infer_device)
                set_loadmodel_depth(infer_model.name)
        elif infer_model.name == "Stable_Diffusion_1_5_controlnet_depth":
                print("IN INFER MODEL FOR CONTROLNET DEPTH ",model_path)
                infer_device_unet = generated_args['infer_device_unet'].name
                infer_device = [infer_device_text, infer_device_unet, infer_device_vae]
                f  = gen.load_models_depth(model_path,infer_device)
                set_loadmodel_depth(infer_model.name)

                #self.Models_loaded_depth = True
           
        elif infer_model.name == "Stable_Diffusion_1_5_int8":
                
                infer_device_unet_pos = generated_args['infer_device_unet_pos'].name
                infer_device_unet_neg = generated_args['infer_device_unet_neg'].name
                infer_device = [infer_device_text, infer_device_unet_pos, infer_device_unet_neg, infer_device_vae]
                f = gen.load_models_int8(model_path,infer_device)
                set_loadmodel(infer_model.name)
        else:
                infer_device_unet = generated_args['infer_device_unet'].name
                infer_device = [infer_device_text, infer_device_unet, infer_device_vae]
                f = gen.load_models(model_path,infer_device)
                set_loadmodel(infer_model.name)
            
  


        gen._active_generation_future = f
        f.call_done_on_exception = False
        #f.add_response_callback(step_callback)
        f.add_exception_callback(exception_callback)
        f.add_done_callback(done_callback)        

   
        return {'FINISHED'}
    
class ProjectLoadModel(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project_load_models"
    bl_label = "Project Load Model"
    bl_description = "Load and compile the SD models"
    bl_options = {'REGISTER'}
    Models_loaded_depth_project_512 = False
    Models_loaded_depth_project_512_int8 = False


    @classmethod
    def poll(cls, context):
        try:

            generated_args = context.scene.dream_textures_project_prompt.generate_args()
            if generated_args["infer_model"].name == "Stable_Diffusion_1_5_controlnet_depth_int8":
       
                context.scene.dream_textures_project_prompt.validate(context, task=ModelType.DEPTH_INT8)
            else:
                context.scene.dream_textures_project_prompt.validate(context, task=ModelType.DEPTH)
                
        except:
            return False
        return Generator.shared().can_use()    

    def execute(self, context):
        scene = context.scene

        

        generated_args = scene.dream_textures_project_prompt.generate_args()
        weight_path = scene.dream_textures_prompt.weight_path
        infer_model = generated_args['infer_model']
        infer_model_size = generated_args['infer_model_size'].name
  

        #print("PROCESS ID in Load model", os.getpid())

        infer_device_text = generated_args['infer_device_text'].name
        

        infer_device_vae = generated_args['infer_device_vae'].name   

        model_path = Path(weight_path) / infer_model.name / infer_model_size
        self.report({'INFO'}, infer_model.name)
        



        def exception_callback(_, exception):
            scene.dream_textures_info = ""
            scene.dream_textures_progress = 0
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(exception))
            raise exception
        
        #global Models_loaded,Models_loaded_depth
        def done_callback(future):
            print("Stable diffusion OpenVino Model compiled and loaded")
          
            set_loadmodel_depth_project(infer_model.name)
          
        try :
            Generator.shared_close()
            kill_loadmodels()
        except:
            pass
        
        gen = Generator.shared()

        #print("PROCESS ID in gen.prompt_to_image in load", os.getpid())

        
        
        if infer_model.name == "Stable_Diffusion_1_5_controlnet_depth_int8":  #"Stable_Diffusion_2_1_depth":
                infer_device_unet_pos = generated_args['infer_device_unet_pos'].name
                infer_device_unet_neg = generated_args['infer_device_unet_neg'].name           
                print("IN INFER MODEL FOR PROJECT CONTROLNET DEPTH",model_path)
                infer_device = [infer_device_text, infer_device_unet_pos, infer_device_unet_neg, infer_device_vae]
           
                f  = gen.load_models_depth_int8(model_path,infer_device)
            
        elif infer_model.name == "Stable_Diffusion_1_5_controlnet_depth":
                infer_device_unet = generated_args['infer_device_unet'].name
                infer_device = [infer_device_text, infer_device_unet, infer_device_vae]
                f  = gen.load_models_depth(model_path,infer_device)
              
          

        gen._active_generation_future = f
        f.call_done_on_exception = False
        #f.add_response_callback(step_callback)
        f.add_exception_callback(exception_callback)
        f.add_done_callback(done_callback)        

   
        return {'FINISHED'}

def set_loadmodel_depth_project(infer_model_size):
    if infer_model_size == 'Stable_Diffusion_1_5_controlnet_depth':

        ProjectLoadModel.Models_loaded_depth_project_512 = True
        ProjectLoadModel.Models_loaded_depth_project_512_int8 = False
        #print("In function set Project: LoadModel.Models_loaded_depth_project_768",ProjectLoadModel.Models_loaded_depth_project_768)
    elif infer_model_size == 'Stable_Diffusion_1_5_controlnet_depth_int8':
        ProjectLoadModel.Models_loaded_depth_project_512_int8 = True
        ProjectLoadModel.Models_loaded_depth_project_512 = False
        #print("In function set Project: LoadModel.Models_loaded_768",ProjectLoadModel.Models_loaded_depth_project_768)

   
    
    #return True
   # print("In function set: ProjectLoadModel.Models_loaded_depth",ProjectLoadModel.Models_loaded_depth_project)
   
def set_loadmodel_depth(infer_model_name):
    if infer_model_name == 'Stable_Diffusion_1_5_controlnet_depth':
        #print("In depth model_size_512 ")
        LoadModel.Models_loaded_depth_512 = True
        LoadModel.Models_loaded_depth_512_int8 = False
        LoadModel.Models_loaded_512 = False
        LoadModel.Models_loaded_512_int8 = False
    elif infer_model_name == 'Stable_Diffusion_1_5_controlnet_depth_int8':
        #print("In depth model_size_768 ")
        LoadModel.Models_loaded_depth_512 = False
        LoadModel.Models_loaded_depth_512_int8 = True
        LoadModel.Models_loaded_512 = False
        LoadModel.Models_loaded_512_int8 = False
    
    #print(" LoadModel.Models_loaded_depth OUTSIDE in dream texture---", LoadModel.Models_loaded_depth_768)
    
    #return True
    

def set_loadmodel(infer_model_name):
    if infer_model_name == 'Stable_Diffusion_1_5':
        #print("In  model_size_512 ")
        LoadModel.Models_loaded_512 = True
        LoadModel.Models_loaded_depth_512_int8 = False
        LoadModel.Models_loaded_depth_512 = False
        LoadModel.Models_loaded_depth_512_int8 = False     

    elif infer_model_name == 'Stable_Diffusion_1_5_int8':
        #print("In  model_size_768 ")
        LoadModel.Models_loaded_512 = False
        LoadModel.Models_loaded_512_int8 = True
        LoadModel.Models_loaded_depth_512 = False
        LoadModel.Models_loaded_depth_512_int8 = False        

    #print(" LoadModel.Models_loaded", LoadModel.Models_loaded_768)

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        generated_args = context.scene.dream_textures_prompt.generate_args()

        try:
      
            context.scene.dream_textures_prompt.validate(context)
            #global Models_loaded,Models_loaded_depth
           # print("C. LoadModel.Models_loaded_depth---", LoadModel.Models_loaded_depth)
           # print("D.  LoadModel.Models_loaded---", LoadModel.Models_loaded)
           # print("E.  ProjectLoadModel.Models_loaded_depth_project in Dream---", ProjectLoadModel.Models_loaded_depth_project)
           
            if generated_args["infer_model"].name == "Stable_Diffusion_1_5" and LoadModel.Models_loaded_512 == True:
                    #print("In generate Models_loaded_512", LoadModel.Models_loaded_512)
                    
                    pass  
            elif generated_args["infer_model"].name == "Stable_Diffusion_1_5_int8" and LoadModel.Models_loaded_512_int8 == True:
                    #print("In generate Models_loaded_768", LoadModel.Models_loaded_768)
                    pass  
            
            
            elif generated_args["infer_model"].name == "Stable_Diffusion_1_5_controlnet_depth":
                if LoadModel.Models_loaded_depth_512 == True or ProjectLoadModel.Models_loaded_depth_project_512 == True:
                    #print("------In generate Models_loaded_depth_512----", LoadModel.Models_loaded_depth_512)
                    pass 
                else:
                    raise ValueError("Please Load the models")                 
            elif generated_args["infer_model"].name == "Stable_Diffusion_1_5_controlnet_depth_int8":
                    if LoadModel.Models_loaded_depth_512_int8 == True or ProjectLoadModel.Models_loaded_depth_project_512_int8 == True:
                        #print("------In generate Models_loaded_depth_768----", LoadModel.Models_loaded_depth_768)
                        pass 
                    else:
                        raise ValueError("Please Load the models") 
            else:
                raise ValueError("Please Load the models")


        except:
            return False
  
        return Generator.shared().can_use()

    def execute(self, context):

                    #print("PROCESS ID in generate", os.getpid())

                    history_template = {prop: getattr(context.scene.dream_textures_prompt, prop) for prop in context.scene.dream_textures_prompt.__annotations__.keys()}
                    history_template["iterations"] = 1
                    history_template["random_seed"] = False
                    is_file_batch = context.scene.dream_textures_prompt.prompt_structure == file_batch_structure.id
                    file_batch_lines = []
                    file_batch_lines_negative = []
                    if is_file_batch:
                        context.scene.dream_textures_prompt.iterations = 1
                        file_batch_lines = [line.body for line in context.scene.dream_textures_prompt_file.lines if len(line.body.strip()) > 0]
                        file_batch_lines_negative = [""] * len(file_batch_lines)
                        history_template["prompt_structure"] = custom_structure.id

                    node_tree = context.material.node_tree if hasattr(context, 'material') and hasattr(context.material, 'node_tree') else None
                    node_tree_center = np.array(node_tree.view_center) if node_tree is not None else None
                    screen = context.screen
                    scene = context.scene

                    generated_args = scene.dream_textures_prompt.generate_args()
                    #context.scene.seamless_result.update_args(generated_args)
                    #context.scene.seamless_result.update_args(history_template, as_id=True)

                    init_image = None
                    if generated_args['use_init_img']:
                        match generated_args['init_img_src']:
                            case 'file':
                                init_image = scene.init_img
                            case 'open_editor':
                                for area in screen.areas:
                                    if area.type == 'IMAGE_EDITOR':
                                        if area.spaces.active.image is not None:
                                            init_image = area.spaces.active.image
                    if init_image is not None:
                        init_image = np.flipud(
                            (np.array(init_image.pixels) * 255)
                                .astype(np.uint8)
                                .reshape((init_image.size[1], init_image.size[0], init_image.channels))
                        )

                    # Setup the progress indicator
                    def step_progress_update(self, context):
                        if hasattr(context.area, "regions"):
                            for region in context.area.regions:
                                if region.type == "UI":
                                    region.tag_redraw()
                        return None
                    bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=generated_args['steps'], update=step_progress_update)
                    scene.dream_textures_info = "Starting..."

                    last_data_block = None
                    def step_callback(_, step_image: ImageGenerationResult):
                        nonlocal last_data_block
                        if step_image.final:
                            return
                        scene.dream_textures_progress = step_image.step
                        if len(step_image.images) > 0:
                            image = step_image.tile_images()
                            last_data_block = bpy_image(f"Step {step_image.step}/{generated_args['steps']}", image.shape[1], image.shape[0], image.ravel(), last_data_block)
                            for area in screen.areas:
                                if area.type == 'IMAGE_EDITOR':
                                    area.spaces.active.image = last_data_block

                    iteration = 0
                    iteration_limit = len(file_batch_lines) if is_file_batch else generated_args['iterations']
                    iteration_square = math.ceil(math.sqrt(iteration_limit))
                    def done_callback(future):
                        nonlocal last_data_block
                        nonlocal iteration
                        if hasattr(gen, '_active_generation_future'):
                            del gen._active_generation_future
                        result: ImageGenerationResult = future.result(last_only=True)
                        for i, result_image in enumerate(result.images):
                            seed = result.seeds[i]
                            prompt_string = context.scene.dream_textures_prompt.prompt_structure_token_subject
                            seed_str_length = len(str(seed))
                            trim_aware_name = (prompt_string[:54 - seed_str_length] + '..') if len(prompt_string) > 54 else prompt_string
                            name_with_trimmed_prompt = f"{trim_aware_name} ({seed})"
                            image = bpy_image(name_with_trimmed_prompt, result_image.shape[1], result_image.shape[0], result_image.ravel(), last_data_block)
                            last_data_block = None
                            if node_tree is not None:
                                nodes = node_tree.nodes
                                texture_node = nodes.new("ShaderNodeTexImage")
                                texture_node.image = image
                                texture_node.location = node_tree_center + ((iteration % iteration_square) * 260, -(iteration // iteration_square) * 297)
                                nodes.active = texture_node
                            for area in screen.areas:
                                if area.type == 'IMAGE_EDITOR':
                                    area.spaces.active.image = image
                            scene.dream_textures_prompt.seed = str(seed) # update property in case seed was sourced randomly or from hash
                            # create a hash from the Blender image datablock to use as unique ID of said image and store it in the prompt history
                            # and as custom property of the image. Needs to be a string because the int from the hash function is too large
                            image_hash = hashlib.sha256((np.array(image.pixels) * 255).tobytes()).hexdigest()
                            image['dream_textures_hash'] = image_hash
                            scene.dream_textures_prompt.hash = image_hash
                            history_entry = context.scene.dream_textures_history.add()
                            for key, value in history_template.items():
                                setattr(history_entry, key, value)
                            history_entry.seed = str(seed)
                            history_entry.hash = image_hash
                            if is_file_batch:
                                history_entry.prompt_structure_token_subject = file_batch_lines[iteration]
                            iteration += 1
                        if iteration < iteration_limit and not future.cancelled:
                            generate_next()
                        else:
                            scene.dream_textures_info = ""
                            scene.dream_textures_progress = 0

                    def exception_callback(_, exception):
                        scene.dream_textures_info = ""
                        scene.dream_textures_progress = 0
                        if hasattr(gen, '_active_generation_future'):
                            del gen._active_generation_future
                        eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(exception))
                        raise exception

                    original_prompt = generated_args["prompt"]
                    original_negative_prompt = generated_args["negative_prompt"]
                    gen = Generator.shared()
                    def generate_next():
                        batch_size = min(generated_args["optimizations"].batch_size, iteration_limit-iteration)
                        if generated_args['pipeline'] == Pipeline.STABILITY_SDK:
                            # Stability SDK is able to accept a list of prompts, but I can
                            # only ever get it to generate multiple of the first one.
                            batch_size = 1
                        if is_file_batch:
                            generated_args["prompt"] = file_batch_lines[iteration: iteration+batch_size]
                            generated_args["negative_prompt"] = file_batch_lines_negative[iteration: iteration+batch_size]
                        else:
                            generated_args["prompt"] = [original_prompt] * batch_size
                            generated_args["negative_prompt"] = [original_negative_prompt] * batch_size
                        supported_model = generated_args['infer_model'].name 
                        def require_depth():
                            print("!!!!!! SUPPORTED MODEL!!!!",supported_model)
                            if supported_model not in ["Stable_Diffusion_1_5_controlnet_depth","Stable_Diffusion_1_5_controlnet_depth_int8"]:
                                self.report({"ERROR"}, "Select a Depth Model")
                                raise ValueError("Selected model does not support depth conditioning. Please select a different model, such as 'Stable_Diffusion_1_5_controlnet_depth' or change the 'Image Type' to 'Color'.")
                        def not_require_depth():  
                             if supported_model in ["Stable_Diffusion_1_5_controlnet_depth","Stable_Diffusion_1_5_controlnet_depth_int8"]:
                                 self.report({"ERROR"}, "Select a non-Depth Model")
                                 raise ValueError("Selected model does not support this pipeline. Please select a different model, such as 'Stable_Diffusion_1_5' or change the 'Image Type' to 'depth'.")

                                                  
                        if init_image is not None:
                            match generated_args['init_img_action']:
                                case 'modify':
                
                                    match generated_args['modify_action_source_type']:
                                        case 'color':
                                            not_require_depth()
                                            if supported_model == "Stable_Diffusion_1_5_int8":
                                                 f = gen.prompt_to_image_int8(
                                                image=init_image,
                                                **generated_args
                                                )
                                            else:
                                                f = gen.prompt_to_image(
                                                    image=init_image,
                                                    **generated_args
                                                )
                                        case 'depth_generated':
                                            require_depth()
                                            if supported_model == "Stable_Diffusion_1_5_controlnet_depth_int8": 
                                                f = gen.depth_to_image_int8(
                                                    image=init_image,
                                                    depth=None,
                                                    **generated_args,
                                                )
                                            else:
                                                f = gen.depth_to_image(
                                                image=init_image,
                                                depth=None,
                                                **generated_args,
                                            )
                                        case 'depth_map':
                                            require_depth()
                                            if supported_model == "Stable_Diffusion_1_5_controlnet_depth_int8": 
                                                f = gen.depth_to_image_int8(
                                                    image=init_image,
                                                    depth=np.array(scene.init_depth.pixels)
                                                            .astype(np.float32)
                                                            .reshape((scene.init_depth.size[1], scene.init_depth.size[0], scene.init_depth.channels)),
                                                    **generated_args,
                                                )
                                            else:
                                                f = gen.depth_to_image(
                                                image=init_image,
                                                depth=np.array(scene.init_depth.pixels)
                                                        .astype(np.float32)
                                                        .reshape((scene.init_depth.size[1], scene.init_depth.size[0], scene.init_depth.channels)),
                                                **generated_args,
                                            )
                                        case 'depth':
                                            require_depth()
                                            if supported_model == "Stable_Diffusion_1_5_controlnet_depth_int8": 
                                                f = gen.depth_to_image_int8(
                                                    image=None,
                                                    depth=np.flipud(init_image.astype(np.float32) / 255.),
                                                    **generated_args,
                                                )
                                            else:
                                                f = gen.depth_to_image(
                                                    image=None,
                                                    depth=np.flipud(init_image.astype(np.float32) / 255.),
                                                    **generated_args,
                                                )                                                
  
                        else:
                            not_require_depth()
                            #print("PROCESS ID in gen.prompt_to_image", os.getpid())
                            
                            if supported_model == "Stable_Diffusion_1_5_int8":
                                f = gen.prompt_to_image_int8(
                                    image=init_image,
                                    **generated_args
                                )
                            else:
                                f = gen.prompt_to_image(
                                    image=init_image,
                                    **generated_args,
                                )
                            #print("TEST TEST TEST !!!!")
                        gen._active_generation_future = f
                        f.call_done_on_exception = False
                        f.add_response_callback(step_callback)
                        f.add_exception_callback(exception_callback)
                        f.add_done_callback(done_callback)
                    generate_next()        


       
                    return {"FINISHED"}

def kill_loadmodels():
        ProjectLoadModel.Models_loaded_depth_project_512 = False
        ProjectLoadModel.Models_loaded_depth_project_512_int8 = False
        LoadModel.Models_loaded_512 = False
        LoadModel.Models_loaded_512_int8 = False
        LoadModel.Models_loaded_depth_512 = False
        LoadModel.Models_loaded_depth_512_int8 = False         
        print("All Models unloaded")

def kill_generator(context=bpy.context):
    Generator.shared_close()
    try:
        context.scene.dream_textures_info = ""
        context.scene.dream_textures_progress = 0
    except:
        pass

class ReleaseGenerator(bpy.types.Operator):
    bl_idname = "shade.dream_textures_release_generator"
    bl_label = "Release Generator"
    bl_description = "Releases the generator class to free up VRAM"
    bl_options = {'REGISTER'}

    def execute(self, context):
        kill_generator(context)
        kill_loadmodels()

        
        return {'FINISHED'}

class CancelGenerator(bpy.types.Operator):
    bl_idname = "shade.dream_textures_stop_generator"
    bl_label = "Cancel Generator"
    bl_description = "Stops the generator without reloading everything next time"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        gen = Generator.shared()
        return hasattr(gen, "_active_generation_future") and gen._active_generation_future is not None and not gen._active_generation_future.cancelled and not gen._active_generation_future.done

    def execute(self, context):
        gen = Generator.shared()
        gen._active_generation_future.cancel()
        context.scene.dream_textures_info = ""
        context.scene.dream_textures_progress = 0
        return {'FINISHED'}
