
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, ReleaseGenerator, CancelGenerator, LoadModel, ProjectLoadModel
from .operators.view_history import SCENE_UL_HistoryList, RecallHistoryEntry, ClearHistory, RemoveHistorySelection, ExportHistorySelection, ImportPromptFile

from .operators.project import ProjectDreamTexture, dream_texture_projection_panels 
from .operators.notify_result import NotifyResult
from .property_groups.dream_prompt import DreamPrompt

from .ui.panels import dream_texture, history

from .ui.presets import DREAM_PT_AdvancedPresets, DREAM_MT_AdvancedPresets, AddAdvancedPreset, RestoreDefaultPresets

CLASSES = (
   
    
    DreamTexture,
    ReleaseGenerator,

    CancelGenerator,
    OpenLatestVersion,
    SCENE_UL_HistoryList,
    RecallHistoryEntry,
    ClearHistory,
    RemoveHistorySelection,
    ExportHistorySelection,
    ImportPromptFile,
 
    ProjectDreamTexture,
    LoadModel,
    ProjectLoadModel,

    DREAM_PT_AdvancedPresets,
    DREAM_MT_AdvancedPresets,
    AddAdvancedPreset,

    NotifyResult,
    
    # The order these are registered in matters
    *dream_texture.dream_texture_panels(),

    *history.history_panels(),
    *dream_texture_projection_panels(),
)

PREFERENCE_CLASSES = (
                      DreamPrompt,
                      RestoreDefaultPresets) 