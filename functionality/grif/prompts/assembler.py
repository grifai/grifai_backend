"""
Assembles the 4-layer system prompt from its components.
Used by config_generator.py at Step 5.
"""

from grif.models.agent_config import PromptLayers
from grif.prompts.layers import load_core_identity, load_role_template


def assemble_system_prompt(
    task_context: str,
    blueprint_id: str,
    user_persona: str = "",
) -> PromptLayers:
    """
    Build all 4 layers and return a PromptLayers object.
    The caller can then call .assemble() to get the final string.

    Layer 1: CoreIdentity — loaded from core_identity.yaml (hardcoded)
    Layer 2: RoleTemplate — loaded from blueprint's template YAML
    Layer 3: TaskContext — provided by Config Generator (LLM-generated)
    Layer 4: UserPersona — from User Profile + Style Guide (optional)
    """
    layer_1 = load_core_identity()
    layer_2 = load_role_template(blueprint_id)
    layer_3 = task_context
    layer_4 = user_persona

    return PromptLayers(
        layer_1_core_identity=layer_1,
        layer_2_role_template=layer_2,
        layer_3_task_context=layer_3,
        layer_4_user_persona=layer_4,
    )


def rebuild_layer_3(
    existing_layers: PromptLayers,
    new_task_context: str,
) -> PromptLayers:
    """
    FORK operation: clone parent prompt layers, replace only Layer 3.
    Used when routing decision is FORK.
    """
    return PromptLayers(
        layer_1_core_identity=existing_layers.layer_1_core_identity,
        layer_2_role_template=existing_layers.layer_2_role_template,
        layer_3_task_context=new_task_context,
        layer_4_user_persona=existing_layers.layer_4_user_persona,
    )
