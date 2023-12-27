

from ..generator_process.models import Pipeline, FixItError
from ..generator_process.actions.huggingface_hub import ModelType
#from ..preferences import OpenURL

def validate(self, context, task: ModelType | None = None) -> bool:

 
   
    if task is None:
        scene = context.scene

        generated_args = scene.dream_textures_prompt.generate_args()
        infer_model = generated_args['infer_model'].name   
        if self.use_init_img:
            match self.init_img_action:
                case 'modify':
                    match self.modify_action_source_type:
                        case 'color':
                            if infer_model == "Stable_Diffusion_1_5_int8":
                                task = ModelType.PROMPT_TO_IMAGE_INT8
                            else:
                                task = ModelType.PROMPT_TO_IMAGE
                        case 'depth_generated' | 'depth_map' | 'depth':
                            if infer_model == "Stable_Diffusion_1_5_controlnet_depth_int8":
                                task = ModelType.DEPTH_INT8
                            else:
                                task = ModelType.DEPTH
        if task is None:
                if infer_model == "Stable_Diffusion_1_5_int8":
                    task = ModelType.PROMPT_TO_IMAGE_INT8
                else:
                    task = ModelType.PROMPT_TO_IMAGE

    # Check if the pipeline supports the task.
    pipeline = Pipeline.STABLE_DIFFUSION #Pipeline[self.pipeline]
    match task:
        case ModelType.DEPTH:
            if not pipeline.depth():
                raise FixItError(
                    f"""The selected pipeline does not support {task.name.replace('_', ' ').lower()} tasks.
Select a different pipeline below.""",
                    lambda _, layout: layout.prop(self, "pipeline")
                )
        case ModelType.DEPTH_INT8:
            if not pipeline.depth():
                raise FixItError(
                    f"""The selected pipeline does not support {task.name.replace('_', ' ').lower()} tasks.
Select a different pipeline below.""",
                    lambda _, layout: layout.prop(self, "pipeline")
                )            

    # Pipeline-specific checks
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            if not Pipeline.local_available():
                raise FixItError(
                    "Local generation is not available for the variant of the add-on you have installed. Choose a different Pipeline such as 'DreamStudio'",
                    lambda _, layout: layout.prop(self, "pipeline")
                )

            if self.infer_model != task.recommended_model():
                    raise FixItError(
                        f"""Incorrect model type selected for {task.name.replace('_', ' ').lower()} tasks.
    Select {task.recommended_model()} for the task from below.""",
                            lambda _, layout: layout.prop(self, "infer_model")

                    )                


    init_image = None
    if self.use_init_img:
        match self.init_img_src:
            case 'file':
                init_image = context.scene.init_img
            case 'open_editor':
                for area in context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        if area.spaces.active.image is not None:
                            init_image = area.spaces.active.image
        if init_image is not None and init_image.type == 'RENDER_RESULT':
            def fix_init_img(ctx, layout):
                layout.prop(self, "init_img_src", expand=True)
                if self.init_img_src == 'file':
                    layout.template_ID(context.scene, "init_img", open="image.open")
                layout.label(text="Or, enable the render pass to generate after each render.")
                #layout.operator(OpenURL.bl_idname, text="Learn More", icon="QUESTION").url = "https://github.com/carson-katri/dream-textures/blob/main/docs/RENDER_PASS.md"
            raise FixItError("""'Render Result' cannot be used as a source image.
Save the image then open the file to use it as a source image.""",
                fix_init_img
            )

    return True