# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

import os

import spacy
from gherkin.parser import Parser
from typing import List, Tuple
from goal_model import *
from utils import *


class Scenario:
    # 一个场景包括多个given, when, then
    def __init__(self, givens_list=None, whens_list=None, thens_list=None, value='', children=None):
        if children is None:
            children = []
        if givens_list is None:
            givens_list = []
        if whens_list is None:
            whens_list = []
        if thens_list is None:
            thens_list = []
        self.givens_list = givens_list
        self.whens_list = whens_list
        self.thens_list = thens_list
        self.value = value
        self.children = children

    def __str__(self):
        return "\n".join([str(given) for given in self.givens_list]) + "\n" + \
               "\n".join([str(when) for when in self.whens_list]) + "\n" + \
               "\n".join([str(then) for then in self.thens_list])


class Background:
    def __init__(self, givens):
        self.givens = givens

    def __str__(self):
        return "Background: " + "\n".join([str(given) for given in self.givens])


class UserStory:
    def __init__(self, as_a, i_want, so_that, scenarios=None, background=None):
        if scenarios is None:
            scenarios: List[Scenario] = []
        self.as_a = as_a
        self.i_want = i_want
        self.so_that = so_that
        self.scenarios = scenarios
        self.background = background

    def __str__(self):
        return "As a " + self.as_a + "\n" + \
               "I want " + self.i_want + "\n" + \
               "So that " + self.so_that + "\n" + \
               str(self.background) + "\n" + \
               "\n".join([str(scenario) for scenario in self.scenarios])


class Feature:
    def __init__(self, feature_name, user_story):
        self.feature_name = feature_name
        self.user_story = user_story

    # 转字符串函数
    def __str__(self):
        return "Feature: " + self.feature_name + "\n" + \
               str(self.user_story)

    # 转json
    def to_json(self):
        return {
            "feature_name": self.feature_name,
            "user_story": {
                "as_a": self.user_story.as_a,
                "i_want": self.user_story.i_want,
                "so_that": self.user_story.so_that,
                "scenarios": [{
                    "givens_list": [given.given for given in scenario.givens_list],
                    "whens_list": [when.when for when in scenario.whens_list],
                    "thens_list": [then.then for then in scenario.thens_list]
                } for scenario in self.user_story.scenarios]
            }
        }


def parse_feature_file(file_path: str) -> Feature:
    with open(file_path, 'r', encoding='utf8') as file:
        feature_text = file.read()
    parser = Parser()
    if 'feature' in parser.parse(feature_text):
        feature_data = parser.parse(feature_text)['feature']
        # print(feature_data)
        feature_name = feature_data['name']
        scenarios = []
        background = None
        for child in feature_data['children']:
            # print(child)
            if 'background' in child:
                temp_background = child['background']
                background = Background([])
                for i in range(0, len(temp_background['steps'])):
                    background.givens.append(temp_background['steps'][i]['text'])
            elif 'scenario' in child:
                givens_list = []
                whens_list = []
                thens_list = []
                scenario = child['scenario']
                value = scenario['name']
                for i in range(0, len(scenario['steps'])):
                    step = scenario['steps'][i]
                    if step['keyword'].strip() == 'Given':
                        givens = [step['text']]
                        while i < len(scenario['steps']) - 1 and scenario['steps'][i + 1]['keyword'].strip() == 'And':
                            givens.append(scenario['steps'][i + 1]['text'])
                            i += 1
                        givens_list.append(givens)
                    elif step['keyword'].strip() == 'When':
                        whens = [step['text']]
                        while i < len(scenario['steps']) - 1 and scenario['steps'][i + 1]['keyword'].strip() == 'And':
                            whens.append(scenario['steps'][i + 1]['text'])
                            i += 1
                        whens_list.append(whens)
                    elif step['keyword'].strip() == 'Then':
                        thens = [step['text']]
                        while i < len(scenario['steps']) - 1 and scenario['steps'][i + 1]['keyword'].strip() == 'And':
                            thens.append(scenario['steps'][i + 1]['text'])
                            i += 1
                        thens_list.append(thens)
                scenarios.append(Scenario(givens_list, whens_list, thens_list, value))
        as_a = ''
        i_want = ''
        so_that = ''
        try:
            as_a, i_want, so_that = convert_user_story(feature_data['description'])
        except:
            print(file_path)
        # as_a = ''
        # i_want = ''
        # so_that = ''
        user_story = UserStory(as_a=as_a, i_want=i_want, so_that=so_that,
                               scenarios=scenarios, background=background)
        return Feature(feature_name=feature_name, user_story=user_story)


def load_program(dir_path: str):
    features = []
    for file in os.listdir(dir_path):
        if file.endswith('.feature'):
            features.append(parse_feature_file(os.path.join(dir_path, file)))
    return features


def convert_user_story(story: str) -> tuple[str, str, str]:
    lines = story.split('\n')
    role = ''
    goal = ''
    benefit = ''
    for line in lines:
        if 'As a' in line or 'As an' in line or 'As the' in line or 'as a' in line or 'as an' in line or 'as the' in line or 'As A' in line or 'As An' in line or 'As The' in line or 'As a' in line or 'As an' in line or 'As the' in line:
            role = line.replace('As a', '').strip()
        elif 'I want to' in line or 'I should be able to' in line or 'I should' in line:
            goal = line.replace('I want to', '').replace('I should be able to', '').strip()
        elif 'In order to' in line or 'So that' in line:
            benefit = line.replace('In order to', '').replace('So that', '').strip()
    return role, goal, benefit


def scenario_to_goal_model(scenario) -> Goal:
    scenario_value = scenario.value
    whens_list = scenario.whens_list
    temp_goal = Goal(value=scenario_value, type='goal')
    if whens_list is not None and len(whens_list) > 0:
        for whens in whens_list:
            if len(whens) > 1:
                for when in whens:
                    goal = Goal(value=when, type='task')
                    temp_goal.add_child(goal)
            else:
                temp_goal.add_child(Goal(value=whens[0], type='task'))
    givens_list = scenario.givens_list
    if givens_list is not None and len(givens_list) > 0:
        for givens in givens_list:
            if len(givens) > 1:
                temp_context = Context(value='temp', type='statement')
                for given in givens:
                    context = Context(value=given, type='fact')
                    temp_context.add_child(context)
                temp_goal.context = temp_context
            else:
                temp_goal.context = Context(value=givens[0], type='fact')
    scenario_children = scenario.children
    if scenario_children is not None and len(scenario_children) > 0:
        for scenario_child in scenario_children:
            temp_goal.add_child(scenario_to_goal_model(scenario_child))
    return temp_goal


def feature_to_goal_model(feature: Feature):
    merge_scenarios(feature)
    user_story = feature.user_story
    root_goal = Goal(value=user_story.i_want, type='goal', so_that=user_story.so_that)
    scenarios = user_story.scenarios
    background = user_story.background
    if background is not None:
        if len(background.givens) > 1:
            temp_context = Context(value='temp', type='statement')
            for given in background.givens:
                context = Context(value=given, type='fact')
                temp_context.add_child(context)
            root_goal.context = temp_context
        else:
            root_goal.context = Context(value=background.givens[0], type='fact')
    if scenarios is not None and len(scenarios) > 0:
        top_goals = []
        for scenario in scenarios:
            top_goals.append(scenario_to_goal_model(scenario))
        root_goal.children = top_goals

    print(str(root_goal))
    return root_goal


# 合并场景
def is_similar(c, core, threshold=0.6):
    if cal_sim(c, core) > threshold:
        return True
    else:
        return False


def merge_scenarios(feature: Feature):
    user_story = feature.user_story
    scenarios = user_story.scenarios
    if len(scenarios) > 1:
        core_scenarios = {}
        for scenario in scenarios:
            # subj, verb, obj = get_core(scenario.value)
            # core = f"{subj} {verb} {obj}"
            core = get_core(scenario.value)
            flag = False
            for c in core_scenarios.keys():
                if is_similar(c, core):
                    core_scenarios[c].append(scenario)
                    flag = True
                    break
            if not flag:
                core_scenarios[core] = [scenario]
        new_scenarios = []
        for c in core_scenarios.keys():
            if len(core_scenarios[c]) <= 1:
                new_scenarios.extend(core_scenarios[c])
            else:
                new_scenario = Scenario(value=c)
                for scenario in core_scenarios[c]:
                    new_scenario.children.append(scenario)
                new_scenarios.append(new_scenario)
        user_story.scenarios = new_scenarios
    return feature


def merge_goals_by_so_that(goals: list[Goal]):
    if len(goals) > 1:
        new_goals = []
        goals_by_so_that = {}
        for g in goals:
            if g.so_that is not None and g.so_that != '':
                flag = False
                for so_that in goals_by_so_that.keys():
                    if is_similar(so_that, g.so_that, 0.8):
                        print(f"合并{so_that}和{g.so_that}")
                        goals_by_so_that[so_that].append(g)
                        flag = True
                        break
                if not flag:
                    goals_by_so_that[g.so_that] = [g]
            else:
                new_goals.append(g)
        for so_that in goals_by_so_that.keys():
            if len(goals_by_so_that[so_that]) <= 1:
                new_goals.extend(goals_by_so_that[so_that])
            else:
                new_goal = Goal(value=goals_by_so_that[so_that][0].value, type='goal', so_that=so_that)
                for goal in goals_by_so_that[so_that]:
                    new_goal.children.append(goal)
                new_goals.append(new_goal)
        return new_goals
    else:
        return goals


def get_core(sentence: str):
    doc = nlp("User signs up with invalid data")

    for token in doc:
        print(token.dep_)

    core_parts = []
    for token in doc:
        if token.dep_ in ("nsubj", "ROOT", "dobj"):
            core_parts.append(token.text)

    print(core_parts)


# def get_core(sentence: str):
#     doc = nlp(sentence)
#     subject = ""
#     verb = ""
#     obj = ""
#     has_obj = False
#     has_subj = False
#     passive = False
#     for token in doc:
#         if "subj" in token.dep_:
#             if "pass" in token.dep_:
#                 passive = True
#             if passive:
#                 if not has_obj:
#                     obj = token.text
#                     has_obj = True
#             else:
#                 if not has_subj:
#                     subject = token.text
#                     has_subj = True
#         elif "obj" in token.dep_:
#             if "pass" in token.dep_:
#                 passive = True
#             if passive:
#                 if not has_subj:
#                     subject = token.text
#                     has_subj = True
#             else:
#                 if not has_obj:
#                     obj = token.text
#                     has_obj = True
#         elif "VERB" == token.pos_:
#             verb = token.lemma_
#     for token in doc:
#         print(token.text, token.dep_, token.head.text, token.head.pos_)
#     # print(f"{subject} {verb} {obj}")
#     return subject, verb, obj

# def get_core(sentence:str):
#     doc = nlp(sentence)
#     new_sentence = []
#     for token in doc:
#         print(token.text, token.dep_, token.head.text, token.head.pos_)
#         if token.dep_ not in ["amod", "advmod", "npadvmod"]:
#             new_sentence.append(token.text)
#     return " ".join(new_sentence)


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    # user_story_feature = './user_story_feature'
    # for dir in os.listdir(user_story_feature):
    #     # print(dir)
    #     save_path = os.path.join('./user_story_json', dir)
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     for file in os.listdir(os.path.join(user_story_feature, dir)):
    #         if file.endswith('.feature'):
    #             feature = parse_feature_file(os.path.join(user_story_feature, dir, file))
    #             # 将feature转为json
    #             if feature is not None:
    #                 feature_json = feature.to_json()
    #                 # 将json写入文件
    #                 with open('./user_story_json/' + dir + '/' + file.replace('.feature', '.json'), 'w',
    #                           encoding='utf8') as f:
    #                     json.dump(feature_json, f, indent=4, ensure_ascii=False)

    # test = './test.feature'
    # nlp = spacy.load("en_core_web_sm")
    # feature = parse_feature_file(test)
    # gm = feature_to_goal_model(feature)
    # dot = Digraph(comment='Goal Model')
    # draw_goal_model(gm, dot)
    # dot.render('./result/goal_model.png', view=True)

    print(get_core("User signs up with invalid data"))

    # feature_dir = './user_story_feature/broth-master'
    # features_by_actor = {}
    # goals_by_actor = {}
    # for file in os.listdir(feature_dir):
    #     if file.endswith('.feature'):
    #         feature = parse_feature_file(os.path.join(feature_dir, file))
    #         if feature is not None:
    #             if feature.user_story.as_a not in features_by_actor.keys():
    #                 features_by_actor[feature.user_story.as_a] = []
    #             features_by_actor[feature.user_story.as_a].append(feature)
    # for actor in features_by_actor.keys():
    #     goal = Goal(value=actor, type='goal')
    #     for feature in features_by_actor[actor]:
    #         if actor not in goals_by_actor.keys():
    #             goals_by_actor[actor] = []
    #         goals_by_actor[actor].append(feature_to_goal_model(feature))
    #     # goals_by_actor[actor] = merge_goals_by_so_that(goals_by_actor[actor])
    #     for g in goals_by_actor[actor]:
    #         goal.children.append(g)
    #     dot = Digraph(comment='Goal Model')
    #     draw_goal_model(goal, dot)
    #     dot.render('./result/goal_model_' + actor + '.png', view=True)
