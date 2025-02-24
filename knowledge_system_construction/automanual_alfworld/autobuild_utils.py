import os
import re
import ast
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def define_functions_from_code(code_str, local_scope):
    """
    Parses the given code string and defines functions within the provided local scope,
    ignoring functions that contain only 'pass' or '...'.
    
    Parameters:
    - code_str: A string containing Python code to parse and execute.
    - local_scope: A dictionary representing the local scope where the functions will be defined.
    """
    try:
        parsed_ast = ast.parse(code_str)
    except Exception:
        return

    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.FunctionDef):
            if len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Expr)):
                continue
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename="<ast>", mode="exec"), local_scope)


class Rule_Manager:
    def __init__(self, init_rules=None, save_path="./gpt4_autobuild_run_logs"):
        self.global_history = {}
        self.save_path = os.path.join(save_path, "rule_manager.json")
        self.all_rules = init_rules
        self.manual = None
        self.responds = []

    def load(self, save_path):
        save_path=os.path.join(save_path, "rule_manager.json")
        if os.path.exists(save_path):
            with open(save_path, 'r') as rf:
                self.global_history = json.load(rf)
            for k, v in self.global_history.items():
                if k == "all_rules":
                    self.all_rules = v
                elif k == "cur_epoch":
                    self.cur_epoch = v
                elif k == "manual":
                    self.manual = v
            if "cur_epoch" not in self.global_history:
                self.cur_epoch = max(int(key.split('_')[1]) for key in self.global_history if key.startswith('epoch_'))

    def save(self):
        # save env configs for epoch
        self.global_history["all_rules"] = self.all_rules
        if self.manual is not None:
            self.global_history["manual"] = self.manual
        with open(self.save_path, 'w') as wf:
            json.dump(self.global_history, wf, indent=4)

    def add_epoch_history(self, cur_epoch, task_type, interact_history, env_history, is_success, error_step):
        self.cur_epoch = cur_epoch
        self.global_history[f"epoch_{cur_epoch}"] = {
                'task_type': task_type,
                'interact_history': interact_history,
                'env_history': env_history,
                'is_success': is_success,
                'error_step': error_step,
                'check_rule': False}
        
    def write_rule(self, rule, type, example="", task_id=""):
        idx = len(self.all_rules)
        self.all_rules[f"rule_{idx}"] = {"rule": rule, "type": type, "example": example, "task_id": task_id}
        self.responds.append(f"Added a new rule \"rule_{idx}\".")

    def update_rule(self, rule_id, **kwargs):
        for k,v in kwargs.items():
            pre_v = self.all_rules[rule_id][k]
            if k == "validation_record":
                v = pre_v + " | " + v
                if len(v.split(" | ")) > 3:
                    v = " | ".join(v.split(" | ")[-3:])
                self.all_rules[rule_id][k] = v
            else:
                self.all_rules[rule_id][k] = v
        self.responds.append(f"Updated a rule \"{rule_id}\".")

    def delete_rule(self, rule_id):
        self.all_rules[rule_id]['rule'] = "delete"
        self.responds.append(f"Deleted a rule \"{rule_id}\".")

    def get_interactions(self, epoch_ids, max_get=2):
        if self.cur_epoch == 0:
            self.responds.append("You cannot call this tool at the first epoch.")
            return
        epoch_ids = epoch_ids.replace(" ", "").split(",")[:max_get]
        # get the history of epoch_ids
        output = []
        for epoch_id in epoch_ids:
            output.append("\n".join([k+":\n"+v for k,v in self.global_history[epoch_id]["interact_history"].items()]))
        if len(output)==0:
            self.responds.append(f"{','.join(epoch_ids)} cannot be found in the history.")
        else:
            self.responds.append("\n\n".join(output))

    def stop_generating(self):
        self.responds.append("Stop generating")

    def report(self):
        print("\n".join([s.split('\n')[0] for s in self.responds]))
        report = "\n".join(self.responds)
        self.responds = []
        return report

    def arrange_rules(self):
        # remove deleted rules
        filtered_rules = {k: v for k, v in self.all_rules.items() if v['rule']!="delete"}
        # rearrange remaining rules
        new_keys = sorted(filtered_rules, key=lambda x: int(x.split('_')[1]))
        reordered_rules = {f"rule_{i}": filtered_rules[key] for i, key in enumerate(new_keys)}
        # update notes to make them point to currect rule
        key_mapping = {}
        for old_key in self.all_rules.keys():
            if old_key in filtered_rules:
                new_key = list(reordered_rules.keys())[list(filtered_rules.keys()).index(old_key)]
                key_mapping[old_key] = new_key
            else:
                key_mapping[old_key] = "deleted_rule"
        for rule in reordered_rules.values():
            for old_key, new_key in key_mapping.items():
                rule['rule'] = rule['rule'].replace("Rule", "rule").replace(old_key, new_key)
                rule['example'] = rule['example'].replace("Rule", "rule").replace(old_key, new_key)
        self.all_rules = reordered_rules

    def rule_string(self):
        if self.manual is not None:
            return "The manual of rules:\n" + self.manual
        if len(self.all_rules) == 0:
            return ""
        out = "Currently found rules:\n"
        for rule_id, rule_dict in self.all_rules.items():
            out += f"{rule_id} (type={rule_dict['type']}): {rule_dict['rule']}"
            if len(rule_dict['example'])>0:
                out += f" For example, {rule_dict['example']}"
            out += "\n"
        return out
    
    def define_functions_from_rules(self, local_scope):
        for rule_dict in self.all_rules.values():
            if len(rule_dict['example'])==0:
                continue
            define_functions_from_code(rule_dict['example'], local_scope)


class Skill_Bank:
    def __init__(self, save_path="./gpt4_autobuild_run_logs"):
        self.skill_dict = {}
        self.save_path = os.path.join(save_path, "skill_bank.json")
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    def load(self, save_path):
        save_path=os.path.join(save_path, "skill_bank.json")
        if os.path.exists(save_path):
            with open(save_path, 'r') as rf:
                self.skill_dict = json.load(rf)
        
    def save(self):
        with open(self.save_path, 'w') as wf:
            json.dump(self.skill_dict, wf, indent=4)

    def add_skill(self, task_type, task_name, init_obs, success_code, direct_success=True):
        if task_type not in self.skill_dict or self.skill_dict[task_type]['success']!=1:
            self.skill_dict[task_type] = {'task_name': task_name, 'init_obs': init_obs, 'skill_code': success_code, 'success': int(direct_success)}

    def add_failure(self, task_type, task_name, init_obs, failure_exp):
        if task_type not in self.skill_dict or self.skill_dict[task_type]['success']==-1:
            self.skill_dict[task_type] = {'task_name': task_name, 'init_obs': init_obs, 'skill_code': failure_exp, 'success': -1}

    def get_relevant_skill(self, query_task_type, local_scope):
        # Use embedding similarity to retrieve relevant skills
        data = [k for k,v in self.skill_dict.items() if v['success']>=0]
        if len(data) == 0:
            output = ""
        else:
            self.vectordb = FAISS.from_texts(data, self.embedding)
            self.retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 1})
            skill_name = self.retriever.get_relevant_documents(query_task_type)[0].page_content
            print(skill_name)
            skill_task = self.skill_dict[skill_name]['task_name']
            skill_obs = self.skill_dict[skill_name]['init_obs']
            skill_code = self.skill_dict[skill_name]['skill_code']
            define_functions_from_code(skill_code, local_scope)
            output = f"\nHere is the code for a relevant skill:\n{skill_obs}\nThe task is to: {skill_task}\n```python\n{skill_code}\n```\n"
            if skill_name == query_task_type:
                output += "Please pay close attention to the process and details of this successful code when writing code. Also, be aware of potential randomness (the current environment may differ from this one)."
        if query_task_type in self.skill_dict and self.skill_dict[query_task_type]['success']<0:
            output += f"\nHere is an failure record from a previous task:\nThe task is to: {self.skill_dict[query_task_type]['task_name']}\n{self.skill_dict[query_task_type]['skill_code']}\nPlease also analyze the plan or mistakes in this failure record in your #Rules to consider# section, and consider how you can avoid the same problems in the plans you are about to make."
        return output