
worker_conclusion_prompt = "Please summarize the misunderstandings and mistakes you made, and then organize your code into a code block. You should copy the used parts from your previous code, including helper methods and steps. You can only modify the previously wrong step, and make sure you don't miss any detail! Your code should not assume that objects are in the same position as they currently are (that means including the finding objects process). So that I can use it with similar but not the same scenarios. If you successfully code as I desrived, I will tip you $200."

worker_reflection_prompt = "You failed to complete the task. Now carefully review the trajectory of the executed actions and the corresponding observations, then identify the reasons for the failure. This reason is often your mistake or misunderstanding rather than the environment's error. By carefully comparing with applicable rules and examples, pinpoint where your code deviated from expected standards. If there exist coding errors, specify the segments of the problematic code and elucidate how they contributed to the errors. If you want to write a correction for an error, you must double-check its plausibility! Additionally, determine whether your codes were in strict adherence to the relevant rules and examples. This reflection and documentation will serve as a reminder for completing future tasks. I will tip $200 for a better response."

worker_treesearch_prompt = "The epoch is over. Please methodically document the following: Start with an overview of your thought process: your initial or adjusted plan, the subgoals you completed, and the issues encountered. Then detail all action steps performed by the agent and their feedbacks (clarify location/object names and interaction methods). If you encountered a coding error, please copy the portion of the code that led to the major error, along with the feedback from the environment. This documentation will serve as a reference for completing future tasks. When recording this information, please don't claim definitive causes of errors or suggestions."

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
* Potential Rules: Describe your thoughts about potential rules based on the current trajectory. Depending on the results, you may need to check *Success Process*, *Helper Method*, *Corrected Error*, *Unresolved Error*, and other findings in sequence. Try to keep it general and universal, and avoid explicitly mentioning function names.
* Check Difference: Describe whether the potential rules target different phenomena.
* Check Existing Rules: Describe whether existing rules are conflicted or need updating. 
* Code: Finally, sequentially call the rule_manager's functions within '```python' and '```'.

[Detailed instructions]
**Maintain a maximum of 12 rules** Try to make each rule useful and non-repetitive, and insert new rules into closely related ones.
**Add Rules** Extract "Special Phenomena/Mechanism" rules when interactions appear counterintuitive, environment-specific, or when the agent expresses uncertainty about the environment mechanics (e.g., using "assume..." in the comment). Refrain from making speculative suggestions or guesses. Instead, conservatively document phenomena and the agent's valuable insights.
**Keep new rules targeted and precise.** Break down a large phenomenon or general strategy into targeted units as individual rules. These can later be upgraded or merged into a more general or larger rule.
**Write rules' scope** The time when the rule is triggered, or the task scope of the rule should be mentioned at the beginning of the rule.
**Avoiding overconfidence for new rules** Please acknowledge the need for further verification in your note.
**Express rules in natural language.** Try to use natural language to describe rules and avoid explicitly mentioning function names.

**Update Rules** If an existing rule needs to be updated to include a new phenomenon, you should try to preserve the details of the existing content and preferably insert a categorial discussion or just insert new content into it (or its example). Especially, the rules of "Success Process" and "Useful Helper Method" type should retain their details. Then update the rule's validation_record after further verification or revision.
'''

builder_success_prompt = builder_base_prompt + '''
**Rules for Success** You should extract "Useful Helper Method" and "Success Process":
* For each useful helper function identified: If it is not already included in a rule, create a rule of type "Useful Helper Method" and record its code unchanged in the rule's example section. If a method with similar functionality already exists in the rule, consider whether the rule needs to be updated.
* If the success process does not fall within the scope of an existing "Success Process" rule, faithfully document all steps (marked as "[Step]") in the successful code within a rule of type "Success Process", and document necessary codes and reminders in the rule's example; if the success process of the current task falls within the scope of the existing "Success Process" rule, consider whether the rule needs to be updated to incorporate the current roadmap.

Follow these instructions. I will tip $200 for a better response.
'''

builder_indirect_case1_prompt = builder_base_prompt + '''
**Rules for Success** You should extract "Useful Helper Method" and "Success Process":
* For each useful helper function identified: If it is not already included in a rule, create a rule of type "Useful Helper Method" and record its code unchanged in the rule's example section. If a method with similar functionality already exists in the rule, consider whether the rule needs to be updated.
* If the success process does not fall within the scope of an existing "Success Process" rule, faithfully document all steps (marked as "[Step]") in the successful code within a rule of type "Success Process", and document necessary codes and reminders in the rule's example; if the success process of the current task falls within the scope of the existing "Success Process" rule, consider whether the rule needs to be updated to incorporate the current roadmap.

**Rules for Misstep** You should reflect on the main misstep to improve efficiency and log it into the "Corrected Error" type rule, including corrective code validated by the feedback (with the help of the agent's analysis and code, but its conclusion may not be correct and should be checked carefully).

Follow these instructions. I will tip $200 for a better response.
'''

builder_indirect_case2_prompt = builder_base_prompt + '''
**Rules for Success** You might need to update "Useful Helper Method" and "Success Process". 
* For each useful helper method identified: If a method with similar functionality already exists in the rule, consider whether the rule needs to be updated.
* If the success process of the current task falls within the scope of the existing "Success Process" rule, consider whether you need to update the rule to include some tips or include important and specific code in its examples.

**Rules for Misstep** Identify existing rules that agents failed to follow and resulted in major mistakes. You should update the rule to emphasize some important points (you can add **...** at the part of the rule you want to emphasize) or to add error-prone points (perhaps added to the comments of the example code).

Follow these instructions. I will tip $200 for a better response.
'''

builder_failure_case1_prompt = builder_base_prompt + '''
**Rules for Final Error** Based on your previous analysis and conclusion, summarize the final error that led to failure. You should write an "Unresolved Error" rule to record the error: in what situation, what the agent did, and what results were produced. So that they can serve as reminders for the agent in the future. Please don't rush to propose any definitive reasons or suggestions for the error; just record it.

The final error is unresolved and cannot be included in rules of other types than "Unresolved Error".
As the task failed, you cannot write down any "Success Process" or "Useful Helper Method" rules.

Follow these instructions. I will tip $200 for a better response.
'''

builder_failure_case2_prompt = builder_base_prompt + '''
**Rules for Misstep** Identify existing rules that agents failed to follow and resulted in major misstep. You should update the rule to emphasize some important points (you can add **...** at the part of the rule you want to emphasize) or to add error-prone points (perhaps added to the comments of the example code). 
Remember that the rules of "Success Process" and "Useful Helper Method" type should retain their details.

Follow these instructions. I will tip $200 for a better response.
'''

builder_merge_get_prompt = '''
Print rule_manager.all_rules:\n{}\n
Warning: Exists beyond {} rules. To streamline the rulebook, try to make each rule useful, non-repetitive and merge rules whose rules or examples can be merged.
Let's merge rules step by step, start by selecting one or two specific rules for consideration. Please decide whether these two rules can to be deleted or merged. Then sequentially call the rule_manager's functions within '```python' and '```'. Please focus on these two rule and refrain from managing other rules. I will tip $200 for a better response.
'''