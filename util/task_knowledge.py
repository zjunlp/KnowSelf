alfworld_put_prompt=f"""When the task requires putting an object on a receptacle, you can follow the guidelines:
Goto(receptacle) -> Open(receptacle)
[Goto(receptacle), Open(receptacle)] -> Take(object, from: receptacle)
Take(object, from: receptacle) -> Goto(receptacle)
[Goto(receptacle), Take(object, from: receptacle), Open(receptacle)] -> Put(object, in/on: receptacle)

Here are the rules you should follow:
Before you open a receptacle, you must first go to it. This rule applies when the receptacle is closed.
To take an object from a receptacle, you either need to be at the receptacle's location, or if it's closed, you need to open it first. 
Before you go to the new receptacle where the object is to be placed, you should take it. 
Putting an object in or on a receptacle can follow either going to the location of the receptacle, taking an object with you, or if receptacle is closed, you need to open it first.

The actions are as follows:
1) go to receptacle 
2) take object from receptacle 
3) put object in/on receptacle
4) open receptacle
"""

alfworld_examine_prompt=f"""When the task requires looking at or examining an object under a desklamp, you can follow the guidelines:
Goto(receptacle) -> Open(receptacle)
[Goto(receptacle), Open(receptacle)] -> Take(object, from: receptacle)
Take(object, from: receptacle) -> Goto(receptacle)
Goto(receptacle) -> Use(receptacle)

Here are the rules you should follow:
Before you open a receptacle, you must first go to it. This rule applies when the receptacle is closed.
To take an object from a receptacle, you either need to be at the receptacle's location, or if it's closed, you need to open it first. 
Before you go to the new receptacle where the object is to be placed, you should take it. 
To use an receptacle, you must go to the place where it is located.

The actions are as follows:
1) go to receptacle 
2) take object from receptacle
3) open receptacle
4) use receptacle 
"""

alfworld_clean_prompt=f"""When the task requires cleaning an object and putting it on a receptacle, you can follow the guidelines:
Goto(receptacle) -> Open(receptacle)
[Goto(receptacle), Open(receptacle)] -> Take(object, from: receptacle)
Take(object, from: receptacle) -> Goto(receptacle)
Goto(receptacle) -> Clean(object, with: receptacle)
[Goto(receptacle), Take(object, from: receptacle), Open(receptacle)] -> Put(object, in/on: receptacle)

Here are the rules you should follow:
Before you open a receptacle, you must first go to it. This rule applies when the receptacle is closed.
To take an object from a receptacle, you either need to be at the receptacle's location, or if it's closed, you need to open it first. 
Before you go to the new receptacle where the object is to be placed, you should take it. 
To clean an object using a receptacle, you must first be at the receptacle's location.
Putting an object in or on a receptacle can follow either going to the location of the receptacle, taking an object with you, or if receptacle is closed, you need to open it first.

The actions are as follows:
1) go to receptacle 
2) take object from receptacle
3) open receptacle
4) clean object with receptacle
5) put object in/on receptacle
"""

alfworld_heat_prompt=f"""When the task requires heating an object and putting it on a receptacle, you can follow the guidelines:
Goto(receptacle) -> Open(receptacle)
[Goto(receptacle), Open(receptacle)] -> Take(object, from: receptacle)
Take(object, from: receptacle) -> Goto(receptacle)
Goto(receptacle) -> Heat(object, with: receptacle)
[Goto(receptacle), Take(object, from: receptacle), Open(receptacle)] -> Put(object, in/on: receptacle)

Here are the rules you should follow:
Before you open a receptacle, you must first go to it. This rule applies when the receptacle is closed.
To take an object from a receptacle, you either need to be at the receptacle's location, or if it's closed, you need to open it first. 
Before you go to the new receptacle where the object is to be placed, you should take it.
To heat an object using a receptacle, you must first be at the receptacle's location.
Putting an object in or on a receptacle can follow either going to the location of the receptacle, taking an object with you, or if receptacle is closed, you need to open it first.

The actions are as follows:
1) go to receptacle 
2) take object from receptacle
3) open receptacle
4) heat object with receptacle
5) put object in/on receptacle 
"""

alfworld_cool_prompt=f"""When the task requires cooling an object and putting it on a receptacle, you can follow the guidelines:
Goto(receptacle) -> Open(receptacle)
[Goto(receptacle), Open(receptacle)] -> Take(object, from: receptacle)
Take(object, from: receptacle) -> Goto(receptacle)
Goto(receptacle) -> Cool(object, with: receptacle)
[Goto(receptacle), Take(object, from: receptacle), Open(receptacle)] -> Put(object, in/on: receptacle)

Here are the rules you should follow:
Before you open a receptacle, you must first go to it. This rule applies when the receptacle is closed.
To take an object from a receptacle, you either need to be at the receptacle's location, or if it's closed, you need to open it first. 
Before you go to the new receptacle where the object is to be placed, you should take it.
To cool an object using a receptacle, you must first be at the receptacle's location.
Putting an object in or on a receptacle can follow either going to the location of the receptacle, taking an object with you, or if receptacle is closed, you need to open it first.

The actions are as follows:
1) go to receptacle 
2) take object from receptacle
3) open receptacle
4) cool object with receptacle
5) put object in/on receptacle 
"""

alfworld_puttwo_prompt=f"""When the task requires putting two objects on a receptacle, you can follow the guidelines:
Goto(receptacle) -> Open(receptacle)
[Goto(receptacle), Open(receptacle)] -> Take(object, from: receptacle)
Take(object, from: receptacle) -> Goto(receptacle)
[Goto(receptacle), Take(object, from: receptacle), Open(receptacle)] -> Put(object, in/on: receptacle)

Here are the rules you should follow:
Before you open a receptacle, you must first go to it. This rule applies when the receptacle is closed.
To take an object from a receptacle, you either need to be at the receptacle's location, or if it's closed, you need to open it first. 
Before you go to the new receptacle where the object is to be placed, you should take it. 
Putting an object in or on a receptacle can follow either going to the location of the receptacle, taking an object with you, or if receptacle is closed, you need to open it first.
Ensure the first object is placed before proceeding to deposit the second object.

The actions are as follows:
1) go to receptacle
2) take object from receptacle
3) open receptacle
4) put object in/on receptacle
"""

alfworld_prompt = {
    "pick_and_place": alfworld_put_prompt,
    "pick_clean_then_place": alfworld_clean_prompt,
    "pick_heat_then_place": alfworld_heat_prompt,
    "pick_cool_then_place": alfworld_cool_prompt,
    "look_at_obj": alfworld_examine_prompt,
    "pick_two_obj": alfworld_puttwo_prompt,
}