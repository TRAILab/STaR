
# def get_scene_prompt(instance_captions: str, start_time: str, end_time: str, 
#                      level_1_names: list, level_2_names: list) -> str:
def get_scene_prompt(instance_captions: str) -> str:
    """
    Generates a structured prompt for hierarchical scene understanding.
    
    Parameters:
        instance_captions (str): Caption text for all detected objects.
        start_time (str): Start time of current video segment.
        end_time (str): End time of current video segment.
        level_1_names (list): History of level 1 area names/descriptions.
        level_2_names (list): History of level 2 area names/descriptions.

    Returns:
        str: Full formatted prompt ready to be passed to the VILA model.
    """
    
    prompt_template = f"""
        <video>\n You are a wandering around a university campus.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific.
        You may also include the following information:
        {instance_captions}
        """
    return prompt_template.strip()


#🎥 Input: Current video (frames from timestamp {start_time} to {end_time})
#   Previously named areas:
# Level 1 history: {', '.join(level_1_names)}
# Level 2 history: {', '.join(level_2_names)}
