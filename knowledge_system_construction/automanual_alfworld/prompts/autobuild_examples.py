init_rules = {
'rule_0': {"rule": "When the task requires cooling, heating or cleaning some object and put them on some receptacle, the agent should try to first search for the target object, then take the object to the receptacle like fridge, microwave or skinbasin to directly cool, heat or clean it, finally go to the target receptacle and put the object.", "type": "Success Process", "example": "", "task_id": "-1"},
'rule_1': {"rule": "When the task requires putting one object on a receptacle, the agent should try to first search for the target object, then take the object, finally go to the target receptacle and put the object.", "type": "Success Process", "example": "", "task_id": "-1"},
'rule_2': {"rule": "When the task requires putting two object on a receptacle, the agent should try to first search for the first target object, then take the object, and go to the target receptacle and put the object. Then the agent should try to find the second target object and repeat the process", "type": "Success Process", "example": "", "task_id": "-1"},
'rule_3': {"rule": "When the task requires looking or examining some object on a desklamp, the agent should try to first search for the target object, then take the object to find where the desklamp is, finally go to the target receptacle and use the desklamp.", "type": "Success Process", "example": "", "task_id": "-1"},
'rule_4': {"rule": "When the agent in a situation that it has taken the target object and need to take other objects, the agent should first go to the designated location and place it, make sure it is holding nothing, and then it can find other objects.", "type": "Error", "example": "", "task_id": "-1"},
'rule_5': {"rule": "When the agent in a situation that it has not found the target object in the current location, the agent should reach other possible locations and should not try other action, such as open some receptacles.", "type": "Error", "example": "", "task_id": "-1"},
'rule_6': {"rule": "When the agent in a situation that it has located the target object, the agent should first take it and should not move to the destination directly.", "type": "Error", "example": "", "task_id": "-1"},
'rule_7': {"rule": "When the agent in a situation that it has reached a closed receptacle and need to take or put objects, the agent should first open the receptacle and then take or put objects.", "type": "Error", "example": "", "task_id": "-1"},
'rule_8': {"rule": "When the agent in a situation that it has taken an object that requires temperature modification to the location, the agent should perform the cooling or heating action directly and should not put the object in it directly.", "type": "Error", "example": "", "task_id": "-1"},
'rule_9': {"rule": "When the agent in a situation that it has taken the target objective and find the desklamp, the agent should use desklamp directly.", "type": "Error", "example": "", "task_id": "-1"}
}

builder_example = '''
Print rule_manager.all_rules: \{...\}

Task ID: 1

Objective:
You are in the middle of a room. Looking quickly around you, you see ... Your task is to: find two pillow and put them in armchair.

Current epoch's trajectory:
Thought: I need to locate two pillows to complete the task. Sofa and bed are common places to find pillows, so I will check them first.
Action: go to sofa 1
Observation: On the sofa 1, you see a pillow 2 and a pillow 1.
Thought: I see two pillows on the sofa. I will take one of them to place in the armchair as instructed.
Action: take pillow 1 from sofa 1
Observation: You pick up the pillow 1 from the sofa 1.

Correct Action and Feedback:
Thought: Now I have taken a pillow, I need to go to armchair to put it.
Action: go to armchair 1
Observation: On the armchair 1, you see nothing.

Explore Action and Feedback:
Thought: I need to locate two pillows to complete the task. Now I have one pillow. I will take another pillow to complete the task.
Action: take pillow 2 from sofa 1
Observation: Nothing happens.

### Outputs ###
In this epoch, the agent is try to complete the task of 'find two pillow and put them in armchair'.

### State before Action:
The agent has successfully located two pillows on the sofa and has taken one of them. The agent is currently holding one pillow and is in the process of trying to take the second pillow from the sofa. The task requires the agent to put both pillows in the armchair.

### Why correct action is correct:
The correct action involves moving to the armchair to place the pillow that the agent is currently holding. This is the next logical step in the process of completing the task, as the agent needs to place the pillow in the armchair.

### Why explore action is incorrect:
The difference bewteen the correct action and the explore action is that explore action attempts to take another pillow (pillow 2) from the same sofa while the agent is already holding one pillow, but the correct action is to move to the armchair to place the pillow. The explore action is not correct because the agent cannot hold two pillows simultaneously. The agent should first place the pillow it is holding in the armchair before attempting to take another pillow like the correct action.

### Potential Rules:
1. #if error# *Error*: When the agent in a situation that it has taken the target object and need to take other objects, the agent should first place it at the designated location, make sure it is holding nothing, and then it can take other objects.
2. #if correct action successfully complete the task# *Success Process*: When the task requires ..., it should first ... This can ensure the task is completed successfully.
3. At the initial observation of the environment, the agent can only observe receptacles, such as cabinet 1, countertop 1. The agent needs to go to the receptacle to view objects in or on it, even for open receptacles. This mechanism is unusual and may be useful for planning purposes. (whether the mechanism is related to the existing rule)

### Check Existing Rules: 
* rule_0: ...(is not related to this trajectory). 
* rule_1: ...(is conflicted or need updating).

### Code:
```python
rule_manager.write_rule(
rule = "When the agent in a situation that it has taken the target object and need to take other objects, the agent should first place it at the designated location, make sure it is holding nothing, and then it can take other objects.",
type = "Error",
example = "",
task_id = "1")

rule_manager.stop_generating()
```
'''

worker_example = '''
You are in the middle of a room. Looking quickly around you, you see cabinet_4, cabinet_3, cabinet_2, cabinet_1, countertop_1, garbagecan_1, sinkbasin_2, sinkbasin_1, toilet_2, toilet_1.
Your task is to: find some spraybottle.

### Understanding of the observation: ...
### Rules to consider: ...
### Plan for the task: I need to go to search each receptacle or surface until seeing a spraybottle. Then, I will take the spraybottle and put it on toilet_1.
### Code:
```python
# Define a helper method to search receptacles for the target object
def find_object(agent, recep_to_check, object_name):
    for receptacle in recep_to_check:
        observation = agent.go_to(receptacle)
        # Check if we need to open the receptacle. If we do, open it.
        if 'closed' in observation:
            observation = agent.open(receptacle)
        # Check if the object is in/on the receptacle.
        if object_name in observation:
            object_ids = get_object_with_id(observation, object_name)
            return object_ids, receptacle
    return None, None

# [Step 1] Get a sorted list of receptacles and surfaces to check for a spraybottle. And use 'find_object' method to search
recep_to_check = ['cabinet_1', 'cabinet_2', 'cabinet_3', 'cabinet_4', 'countertop_1', 'toilet_1', 'toilet_2', 'sinkbasin_1', 'sinkbasin_2', 'garbagecan_1']
object_ids, receptacle_with_spraybottle = find_object(agent, recep_to_check, 'spraybottle')
assert object_ids is not None, f'Error in [Step 1]: There is no spraybottle in/on {recep_to_check}.'

# [Step 2] Take the spraybottle
found_spraybottle = object_ids[0]
observation = agent.take_from(found_spraybottle, receptacle_with_spraybottle)
assert agent.holding == found_spraybottle, f'Error in [Step 2]: I cannot take {found_spraybottle} from {receptacle}.'

# [Step 3] Go to a toilet and put the spraybottle on it
observation = agent.go_to('toilet_1')
# check if toilet_1 is closed. If so, open it.
if 'closed' in observation:
    observation = agent.open('toilet_1')
observation = agent.put_in_or_on(found_spraybottle, 'toilet_1')
```

### Feedbacks ###
obs_1: Act: agent.go_to('cabinet_1'). Obs: On cabinet_1, you see cloth_1, soapbar_1, soapbottle_1. You are at cabinet_1 and holding nothing.
obs_2: Act: agent.go_to('cabinet_2'). Obs: cabinet_2 is closed. You are at cabinet_2 and holding nothing.
obs_3: Act: agent.open('cabinet_2'). Obs: You open cabinet_2. In cabinet_2, you see candle_1, and spraybottle_2. You are at cabinet_2 and holding nothing.
obs_4: Act: agent.take_from('spraybottle_2', 'cabinet_2'). Obs: You take spraybottle_2 from cabinet_2. You are at cabinet_2 and holding spraybottle_2.
obs_5: Act: agent.go_to('toilet_1'). Obs: On toilet_1, you see glassbottle_1. You are at toilet_1 and holding spraybottle_2.
obs_6: Act: agent.put_in_or_on('spraybottle_2', 'toilet_1'). Obs: You put spraybottle_2 in/on toilet_1. You are at toilet_1 and holding nothing. This epoch is done. Succeed: True
'''.strip()

formulater_example = '''
Print rule_manager.all_rules: \{'rule_0': ..., 'rule_1': ..., ...\}

### General Understandings: ...

### Category of Rules: ...

Here is the resulting manual:
```markdown
# Manual Name

## Overview

Briefly introduce the topics covered and the main objectives of the rules.

## Category 1 Name

### Introduction

A brief description of the category, what the rules have in common, and highlighting key contents.

#### Included Ruls

- **rule_3**: A short rule description
- **rule_10**: A short rule description

## Category 2 Name

### Introduction

A brief description of the category, what the rules have in common, and highlighting key contents.

#### Included Ruls

- **rule_1**: A short rule description
- **rule_4**: A short rule description
...
```
'''