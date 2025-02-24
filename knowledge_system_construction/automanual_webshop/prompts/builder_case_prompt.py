worker_organize_prompt = "Please organize your code into a code block. You should copy the used parts from your previous code, including all useful steps, and make sure you don't miss any detail! In the comment of your code, use *Pause here* to when to pause reviewing the resulting web page. So that I can follow this roadmap to successfully navigate and complete similar tasks from scratch. If you successfully code as I desrived, I will tip you $200."

worker_conclusion_prompt = "Please summarize the misunderstandings and mistakes you made, and then organize your code into a code block. You should copy the used parts from your previous code, including all useful steps. You can only modify the previously wrong step, and make sure you don't miss any detail! In the comment of your code, use *Pause here* to specify when to pause reviewing the resulting web page. So that I can follow this roadmap to successfully navigate and complete similar tasks from scratch. If you successfully code as I desrived, I will tip you $200."

worker_reflection_prompt = "You failed to complete the task. Now carefully review the trajectory of the executed actions and the corresponding observations, then identify the reasons for the failure. This reason is often your mistake or misunderstanding rather than the environment's error. By carefully comparing with applicable rules and examples, pinpoint where your code deviated from expected standards. If you want to write a correction for an error, you must double-check its plausibility! Additionally, determine whether your codes were in strict adherence to the relevant rules and examples. This reflection and documentation will serve as a reminder for completing future tasks. I will tip $200 for a better response."

indirect_case_classify_prompt = '''
Please analyze the scenario to identify the root cause of the observed mistakes, and describe the existing rules related to the mistakes. Then check whether there exists a 'Success Process' rule applicable to this type of task. Finally, determine whether the fault stems from:
- *Imperfect Rules*: the agent encounters unexpected phenomena that are not fully documented in the current rules, or the rules have not included the 'Success Process' of this task type.
- *Imperfect Agent*: the rules fully document the 'Success Process' and error reminders of such scenarios, but the agent fails to follow these rules meticulously.
Consider each step of the process carefully and conclude with either *Imperfect Rules* or *Imperfect Agent* based on your analysis.
'''

failure_case_classify_prompt = '''
Please begin by methodically evaluating the process: identify which steps were successfully executed by the agent and at which point the final error that led to failure occurred (You do not need to determine the root cause of the failure). Also, describe the existing rules related to each step. Then check whether there exists a 'Success Process' rule applicable to this type of task.

Finally, determine whether the fault lies in:
- *Imperfect Rules*: The agent encounters not anticipated phenomena or fully documented in the current rules, or the rules have not included the 'Success Process' of this task type.
- *Imperfect Agent*: Despite the rules providing a complete documentation of the 'Success Process' and error reminders of such scenarios, the agent fails to follow these rules meticulously.
Conclude your analysis by categorizing either *Imperfect Rules* or *Imperfect Agent*.
'''

builder_base_prompt = '''
[Output Format Instructions]
Based on the current trajectory and your analysis, you should output the following things:
* Potential Rules: Describe your thoughts about potential rules based on the current trajectory. Depending on the results, you may need to check *Success Process*, *Corrected Error*, *Unresolved Error*, and other findings in sequence. Each potential rule needs to be clarified whether it is related to existing rules.
* Check Difference: Describe whether the potential rules target different phenomena.
* Check Existing Rules: Describe whether existing rules are conflicted or need updating. 
* Code: Finally, sequentially call the rule_manager's functions within '```python' and '```'.

[Detailed instructions]
**Maintain a maximum of 15 rules** Try to make each rule useful and non-repetitive, and insert new rules into closely related ones.
**Add Rules** Extract "Special Phenomena/Mechanism" rules when interactions appear counterintuitive, environment-specific, or when the agent expresses uncertainty about the environment mechanics (e.g., using "assume..." in the comment). Refrain from making speculative suggestions or guesses. Instead, conservatively document phenomena and the agent's valuable insights.
**Keep new rules targeted and precise.** Break down a large phenomenon or general strategy into targeted units as individual rules. These can later be upgraded or merged into a more general or larger rule.
**Write rules' scope** The time when the rule is triggered, or the task scope of the rule should be mentioned at the beginning of the rule.
**Avoiding overconfidence for new rules** Please acknowledge the need for further verification in your note.

**Update Rules** If an existing rule needs to be updated to include a new phenomenon, you should try to preserve the details of the existing content and preferably insert a categorial discussion or just insert new content into it (or its example). Especially, the rules of "Success Process" type should retain their details. Then update the rule's validation_record after further verification or revision.
'''

builder_success_prompt = builder_base_prompt + '''
**Rules for Success** You should extract "Success Process":
* If the success process does not fall within the scope of an existing "Success Process" rule, faithfully document all steps (marked as "[Step]") in the successful code within a rule of type "Success Process", and document necessary codes and reminders in the rule's example; if the success process of the current task falls within the scope of the existing "Success Process" rule, consider whether the rule needs to be updated to incorporate the current roadmap.

Follow these instructions. I will tip $200 for a better response.
'''

builder_indirect_case1_prompt = builder_base_prompt + '''
**Rules for Success** You should extract "Success Process":
* If the success process does not fall within the scope of an existing "Success Process" rule, faithfully document all steps (marked as "[Step]") in the successful code within a rule of type "Success Process", and document necessary codes and reminders in the rule's example; if the success process of the current task falls within the scope of the existing "Success Process" rule, consider whether the rule needs to be updated to incorporate the current roadmap.

**Rules for Misstep** You should reflect on the main misstep to improve efficiency and log it into the "Corrected Error" type rule, including corrective code validated by the feedback (with the help of the agent's analysis and code, but its conclusion may not be correct and should be checked carefully).

Follow these instructions. I will tip $200 for a better response.
'''

builder_indirect_case2_prompt = builder_base_prompt + '''
**Rules for Success** You might need to update "Success Process". 
* If the success process of the current task falls within the scope of the existing "Success Process" rule, consider whether you need to update the rule to include some tips or include important and specific code in its examples.

**Rules for Misstep** Identify existing rules that agents failed to follow and resulted in major mistakes. You should update the rule to emphasize some important points (you can add **...** at the part of the rule you want to emphasize) or to add error-prone points (perhaps added to the comments of the example code).

Follow these instructions. I will tip $200 for a better response.
'''

builder_failure_case1_prompt = builder_base_prompt + '''
**Rules for Final Error** Based on your previous analysis and conclusion, summarize the final error that led to failure. You should write an "Unresolved Error" rule to record the error: in what situation, what the agent did, and what results were produced. So that they can serve as reminders for the agent in the future. Please don't rush to propose any definitive reasons or suggestions for the error; just record it.

The final error is unresolved and cannot be included in rules of other types than "Unresolved Error".
As the task failed, you cannot write down any "Success Process" rules.

Follow these instructions. I will tip $200 for a better response.
'''

builder_failure_case2_prompt = builder_base_prompt + '''
**Rules for Misstep** Identify existing rules that agents failed to follow and resulted in major misstep. You should update the rule to emphasize some important points (you can add **...** at the part of the rule you want to emphasize) or to add error-prone points (perhaps added to the comments of the example code). 
Remember that the rules of "Success Process" type should retain their details.

Follow these instructions. I will tip $200 for a better response.
'''

builder_merge_get_prompt = '''
Print rule_manager.all_rules:\n{}\n
Warning: Exists beyond {} rules. To streamline the rulebook, try to make each rule useful, non-repetitive and merge rules whose rules or examples can be merged.
Let's merge rules step by step, start by selecting one or two specific rules for consideration. Please decide whether these two rules can to be deleted or merged. Then sequentially call the rule_manager's functions within '```python' and '```'. Please focus on these two rule and refrain from managing other rules. I will tip $200 for a better response.
'''

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