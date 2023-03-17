# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

import os
from gherkin.parser import Parser
from typing import List, Tuple


class Givens:
    def __init__(self, given: list = None):
        if given is None:
            given = []
        self.given = given

    def append(self, given):
        self.given.append(given)

    def __str__(self):
        return "\nGiven " + "\nAnd ".join(self.given) + "\n"


class Whens:
    def __init__(self, when: list = None):
        if when is None:
            when = []
        self.when = when

    def append(self, when):
        self.when.append(when)

    def __str__(self):
        return "\nWhen " + "\nAnd ".join(self.when) + "\n"


class Thens:
    def __init__(self, then: list = None):
        if then is None:
            then = []
        self.then = then

    def append(self, then):
        self.then.append(then)

    def __str__(self):
        return "\nThen " + "\nAnd ".join(self.then) + "\n"


class Scenario:
    # 一个场景包括多个given, when, then
    def __init__(self, givens_list=None, whens_list=None, thens_list=None):
        if givens_list is None:
            givens_list: List[Givens] = []
        if whens_list is None:
            whens_list: List[Whens] = []
        if thens_list is None:
            thens_list: List[Thens] = []
        self.givens_list = givens_list
        self.whens_list = whens_list
        self.thens_list = thens_list

    def __str__(self):
        return "\n".join([str(given) for given in self.givens_list]) + "\n" + \
               "\n".join([str(when) for when in self.whens_list]) + "\n" + \
               "\n".join([str(then) for then in self.thens_list])


class UserStory:
    def __init__(self, as_a, i_want, so_that, scenarios=None):
        if scenarios is None:
            scenarios: List[Scenario] = []
        self.as_a = as_a
        self.i_want = i_want
        self.so_that = so_that
        self.scenarios = scenarios

    def __str__(self):
        return "As a " + self.as_a + "\n" + \
               "I want " + self.i_want + "\n" + \
               "So that " + self.so_that + "\n" + \
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
        for child in feature_data['children']:
            # print(child)
            if 'scenario' in child:
                givens_list = []
                whens_list = []
                thens_list = []
                scenario = child['scenario']
                for i in range(0, len(scenario['steps'])):
                    step = scenario['steps'][i]
                    if step['keyword'].strip() == 'Given':
                        givens = Givens([step['text']])
                        while i < len(scenario['steps']) - 1 and scenario['steps'][i + 1]['keyword'].strip() == 'And':
                            givens.append(scenario['steps'][i + 1]['text'])
                            i += 1
                        givens_list.append(givens)
                    elif step['keyword'].strip() == 'When':
                        whens = Whens([step['text']])
                        while i < len(scenario['steps']) - 1 and scenario['steps'][i + 1]['keyword'].strip() == 'And':
                            whens.append(scenario['steps'][i + 1]['text'])
                            i += 1
                        whens_list.append(whens)
                    elif step['keyword'].strip() == 'Then':
                        thens = Thens([step['text']])
                        while i < len(scenario['steps']) - 1 and scenario['steps'][i + 1]['keyword'].strip() == 'And':
                            thens.append(scenario['steps'][i + 1]['text'])
                            i += 1
                        thens_list.append(thens)
                scenarios.append(Scenario(givens_list, whens_list, thens_list))
        as_a = ''
        i_want = ''
        so_that = ''
        try:
            # if 'So that' in feature_data['description'] or 'so that' in feature_data['description']:
            #     as_a = feature_data['description'].split(['As a', 'As an'])[1].split('I want ')[0].strip()
            #     i_want = feature_data['description'].split('I want ')[1].split(['so that', 'So that'])[0].strip()
            #     so_that = feature_data['description'].split(['so that', 'So that'])[1].strip()
            # elif 'In order to' in feature_data['description'] or 'in order to' in feature_data['description']:
            #     so_that = feature_data['description'].split(['In order to', 'in order to'])[1].split('As a')[0].strip()
            #     as_a = feature_data['description'].split(['As a', 'As an'])[1].split('I want ')[0].strip()
            #     i_want = feature_data['description'].split('I want ')[1].strip()
            as_a, i_want, so_that = convert_user_story(feature_data['description'])
        except:
            print(file_path)
        # as_a = ''
        # i_want = ''
        # so_that = ''
        user_story = UserStory(as_a=as_a, i_want=i_want, so_that=so_that,
                               scenarios=scenarios)
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


if __name__ == '__main__':
    user_story_feature = './user_story_feature'
    for dir in os.listdir(user_story_feature):
        # print(dir)
        save_path = os.path.join('./user_story_json', dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for file in os.listdir(os.path.join(user_story_feature, dir)):
            if file.endswith('.feature'):
                feature = parse_feature_file(os.path.join(user_story_feature, dir, file))
                # 将feature转为json
                if feature is not None:
                    feature_json = feature.to_json()
                    # 将json写入文件
                    with open('./user_story_json/' + dir + '/' + file.replace('.feature', '.json'), 'w',
                              encoding='utf8') as f:
                        json.dump(feature_json, f, indent=4, ensure_ascii=False)
