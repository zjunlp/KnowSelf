import re
import json
import logging
from typing import Any, Dict, List, Tuple

from colorama import Fore
from collections import Counter

logger = logging.getLogger("agent_frame")

action_list = [
    "go to",
    "take",
    "put",
    "open",
    "close",
    "toggle",
    "clean",
    "heat",
    "cool",
    "use",
    "examine"
]

class Action:
    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attr_list = []
        for key in self.__dict__.keys():
            if getattr(self, key) is not None:
                value = getattr(self, key)
                if isinstance(value, str):
                    value = f'"{value}"'
                attr_list.append(f"{key}={value}")
        attr = ', '.join(attr_list)
        return f"Action({attr})"

class Goto(Action):
    def __init__(self, destination, status="normal", seen_objs=[], **kwargs):
        super().__init__("go to", **kwargs)
        self.destination = destination
        self.status = status
        self.seen_objs = seen_objs

class Take(Action):
    def __init__(self, obj, source, **kwargs):
        super().__init__("take", **kwargs)
        self.obj = obj
        self.source = source

class Put(Action):
    def __init__(self, obj, destination, **kwargs):
        super().__init__("put", **kwargs)
        self.obj = obj
        self.destination = destination
        
class Open(Action):
    def __init__(self, obj, seen_objs=[], **kwargs):
        super().__init__("open", **kwargs)
        self.obj = obj
        self.seen_objs = seen_objs
        
class Close(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("close", **kwargs)
        self.obj = obj

class Toggle(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("toggle", **kwargs)
        self.obj = obj

class Clean(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("clean", **kwargs)
        self.obj = obj

class Heat(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("heat", **kwargs)
        self.obj = obj
        
class Cool(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("cool", **kwargs)
        self.obj = obj

class Use(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("use", **kwargs)
        self.obj = obj
        
class Examine(Action):
    def __init__(self, obj, **kwargs):
        super().__init__("examine", **kwargs)
        self.obj = obj

class AlfworldState:
    def __init__(self, **kwargs) -> None:
        self.loc = None
        self.reachable_locs = []
        self.obj_in_hand = None
        self.objs_in_loc = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self):
        attr_list = []
        for key in self.__dict__.keys():
            if getattr(self, key) is not None:
                value = getattr(self, key)
                if isinstance(value, str):
                    value = f'"{value}"'
                attr_list.append(f"{key}={value}")
        attr = ', '.join(attr_list)
        return f"AlfworldState({attr})"
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AlfworldState):
            return False
        return Counter(self.loc) == Counter(value.loc) and Counter(self.obj_in_hand) == Counter(value.obj_in_hand)
        
    def parse_reachable_locs(self, task_desc):
        task_desc = task_desc.strip()
        pattern = r'(a|an)\s+(\w+)\s+(\d+)' # a/an <object> <num>
        matchs = re.findall(pattern, task_desc)
        for _, obj, num in matchs:
            self.reachable_locs.append(f"{obj} {num}")
        
    def parse_action(self, action: str, observation: str) -> Action:
        action = action.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        if re.findall(pattern, action) == []:
            return None
        action = re.findall(pattern, action)[0] # get first action
        assert action is not None
        punctuations = [".", "!", "?", ",", ";", ":"]
        while action and action[-1] in punctuations:
            action = action[:-1]
        
        observation = observation.strip()
        pattern = re.compile(r"Observation:\s?(.*)", re.DOTALL)
        observation = re.findall(pattern, observation)[0] # get first observation
        # assert observation is not None
        if "Nothing happens" in observation and "go to" not in action:
            return None

        objs = []
        if "you see" in observation:
            s = observation.split("you see")[1]
            pattern = r'(a|an)\s+([\w]+(?:\s+[\w]+)*?)\s+(\d+)' # # a/an <object> <num>
            matchs = re.findall(pattern, s)
            for _, obj, num in matchs:
                objs.append(f"{obj} {num}")
        
        for action_name in action_list:
            if action_name in action:
                if action_name == "go to":
                    destination = re.findall(r"go to (.*)", action)[0]
                    if "open" in observation:
                        return Goto(destination, status="opened", seen_objs=objs)
                    elif "closed" in observation:
                        return Goto(destination, status="closed", seen_objs=objs)
                    return Goto(destination, seen_objs=objs)
                elif action_name == "take":
                    obj, source = re.findall(r"take (.*) from (.*)", action)[0]
                    return Take(obj, source)
                elif action_name == "put":
                    obj, destination = re.findall(r"put (.*) in/on (.*)", action)[0]
                    return Put(obj, destination)
                elif action_name == "open":
                    obj = re.findall(r"open (.*)", action)[0]
                    return Open(obj, seen_objs=objs)
                elif action_name == "close":
                    obj = re.findall(r"close (.*)", action)[0]
                    return Close(obj)
                elif action_name == "toggle":
                    obj = re.findall(r"toggle (.*)", action)[0]
                    return Toggle(obj)
                elif action_name == "clean":
                    obj = re.findall(r"clean (.*) with", action)[0]
                    return Clean(obj)
                elif action_name == "heat":
                    obj = re.findall(r"heat (.*) with", action)[0]
                    return Heat(obj)
                elif action_name == "cool":
                    obj = re.findall(r"cool (.*) with", action)[0]
                    return Cool(obj)
                elif action_name == "use":
                    obj = re.findall(r"use (.*)", action)[0]
                    return Use(obj)
                elif action_name == "examine":
                    obj = re.findall(r"examine (.*)", action)[0]
                    return Examine(obj)
                
        logger.info(f"\n{Fore.RED}Action can't be parsed from output{Fore.RESET}")
        return None
        
    
    def transition(self, action):
        if action is None:
            raise ValueError("Action is None")
        if action.name == "go to":
            if action.destination in self.reachable_locs:
                self.loc = {
                    "name": action.destination,
                    "status": action.status
                }
                if action.seen_objs:
                    if self.objs_in_loc is None:
                        self.objs_in_loc = {}
                    if action.destination in self.objs_in_loc:
                        self.objs_in_loc[action.destination].extend(action.seen_objs)
                        self.objs_in_loc[action.destination] = list(set(self.objs_in_loc[action.destination]))
                    else:
                        self.objs_in_loc[action.destination] = action.seen_objs
            else:
                raise ValueError(f"Goto Error: Destination {action.destination} is not reachable from {self.loc}")
        
        elif action.name == "take":
            if self.obj_in_hand is not None:
                raise ValueError("Take Error: Already holding an object")
            self.obj_in_hand = {
                "name": action.obj,
                "status": "normal"
            }
            if action.source in self.objs_in_loc:
                self.objs_in_loc[action.source].remove(action.obj)
                
        elif action.name == "put":
            if self.obj_in_hand is None:
                raise ValueError("Put Error: No object in hand to put")
            if action.destination in self.objs_in_loc:
                self.objs_in_loc[action.destination].append(action.obj)
            else:
                self.objs_in_loc[action.destination] = [action.obj]
            self.obj_in_hand = None
            
        elif action.name == "open":
            # only closed locations can be opened
            # if self.loc["status"] != "closed":
            #     raise ValueError("Open Error: Location is not closed")
            if self.loc["name"] != action.obj:
                raise ValueError("Open Error: Location to be opened is not the current location")
            
            self.loc["status"] = "opened"
            if action.seen_objs:
                if self.objs_in_loc is None:
                    self.objs_in_loc = {}
                if self.loc["name"] in self.objs_in_loc:
                    self.objs_in_loc[self.loc["name"]].extend(action.seen_objs)
                    self.objs_in_loc[self.loc["name"]] = list(set(self.objs_in_loc[self.loc["name"]]))
                else:
                    self.objs_in_loc[self.loc["name"]] = action.seen_objs
            
        elif action.name == "close":
            # only opened locations can be closed
            # if self.loc["status"] != "opened":
            #     raise ValueError("Close Error: Location is not opened")
            if self.loc["name"] != action.obj:
                raise ValueError("Close Error: Location to be closed is not the current location")
            self.loc["status"] = "closed"
            pass
        elif action.name == "toggle":
            pass
        
        elif action.name == "clean":
            if self.obj_in_hand is None:
                raise ValueError("Clean Error: No object in hand to clean")
            # if self.obj_in_hand["name"] != action.obj:
            #     raise ValueError("Clean Error: Object to be cleaned is not the object in hand")
            self.obj_in_hand["status"] = "cleaned"
            
        elif action.name == "heat":
            if self.obj_in_hand is None:
                raise ValueError("Heat Error: No object in hand to heat")
            # if self.obj_in_hand["name"] != action.obj:
            #     raise ValueError("Heat Error: Object to be heated is not the object in hand")
            self.obj_in_hand["status"] = "heated"
            
        elif action.name == "cool":
            if self.obj_in_hand is None:
                raise ValueError("Cool Error: No object in hand to cool")
            # if self.obj_in_hand["name"] != action.obj:
            #     raise ValueError("Cool Error: Object to be cooled is not the object in hand")
            self.obj_in_hand["status"] = "cooled"
        
        elif action.name == "use":
            pass
        elif action.name == "examine":
            pass
        else:
            logger.info(f"\n{Fore.RED}Action can't be parsed from output{Fore.RESET}")
            return
    
    def reset(self):
        self.loc = None
        self.reachable_locs = []
        self.obj_in_hand = None
        self.objs_in_loc = {}
        
    def to_json(self):
        return json.dumps(self.__dict__)


        