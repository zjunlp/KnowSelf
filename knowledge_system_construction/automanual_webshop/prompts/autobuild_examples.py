init_rules = {
'rule_0': {"rule": "When the agent in a situation that it has search the product and been in the search result page, the agent should carefully examine the features of each product, and then click on the one that best meets the requirements of the task.", "type": "Error", "example": "", "task_id": "-1"},
'rule_1': {"rule": "When the agent in a situation that it has clicked the product and been in the product page, the agent should carefully examine each feature option of the product, and ensure that it has selected the feature options required by the task like pack, size, color and so on.", "type": "Error", "example": "", "task_id": "-1"},
'rule_2': {"rule": "When the agent in a situation that it has confirmed the product he needs to purchase and ensured that it meets the requirements of the task, it should click the 'Buy Now' button.", "type": "Error", "example": "", "task_id": "-1"},
'rule_3': {"rule": "When the agent in a situation that it has confirmed the current product can't meet the task requirements, it needs to search for another product that meets the criteria instead of proceeding with the current product.", "type": "Error", "example": "", "task_id": "-1"}
}

builder_example = """
Print rule_manager.all_rules: \{...\}

Task ID: 1

Objective:
i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars

Current epoch's trajectory:
Thought: I think I should use the search bar to look for the product I need.
Action: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]
Observation:
Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars ... [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] ...
Thought: I think I should click on the product I need, which is B000VOHH8I.
Action: click[B000VOHH8I]
Observation:
Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now

Correct Action:
Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.
Action: click[6.76 fl oz (pack of 1)]

Explore Action:
Thought: I think I should click on the 'Buy Now' button to purchase the product.
Action: click[Buy Now]

### Outputs ###
In this epoch, the agent is try to complete the task of 'find a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars'.

### State before Action:
The agent has successfully clicked on the product 'B000VOHH8I', which is l'eau d'issey under 100 dollars, and is now observing the product page.

### Why correct action is correct:
The correct action involves selecting the size of the product that meets the task requirements. This action is correct because the agent needs to ensure that the product meets the task requirements before purchasing it.

### Why explore action is incorrect:
The difference bewteen the correct action and the explore action is that explore action attempts to purchase the product without ensuring that it meets the task requirements about the size. This action is incorrect because the agent needs to ensure that the product meets the task requirements before purchasing it. The agent should first select the size of the product that meets the task requirements.

### Potential Rules:
1. #if error# *Error*: When the agent in a situation that it has clicked the product and been in the product page, the agent should carefully examine each feature option of the product, and ensure that it has selected the feature options required by the task like pack, size, color and so on.
2. #if correct action successfully complete the task with [buy now]# *Success Process*: When the task requires ..., it should first ... This can ensure the task is completed successfully.

### Check Existing Rules: 
* rule_0: ...(is not related to this trajectory). 
* rule_1: ...(is conflicted or need updating).

### Code:
```python
rule_manager.write_rule(
rule = "When the agent in a situation that it has clicked the product and been in the product page, the agent should carefully examine each feature option of the product, and ensure that it has selected the feature options required by the task like pack, size, color and so on.",
type = "Error",
example =  '''
...
''',
task_id = "1")

rule_manager.stop_generating()
```
"""