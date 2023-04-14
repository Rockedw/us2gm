# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

import os
import time

import spacy
from gherkin.parser import Parser
from typing import List, Tuple

from nltk.corpus import wordnet

import utils
from goal_model import *
from utils import *

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


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
            if len(givens) > 0:
                temp_context = Context(value='temp', type='statement')
                for given in givens:
                    context = Context(value=given, type='fact')
                    temp_context.add_child(context)
                temp_goal.context = temp_context
            # else:
            #     temp_goal.context = Context(value=givens[0], type='fact')
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
        if len(background.givens) > 0:
            temp_context = Context(value='temp', type='statement')
            for given in background.givens:
                context = Context(value=given, type='fact')
                temp_context.add_child(context)
            root_goal.context = temp_context
        # else:
        #     root_goal.context = Context(value=background.givens[0], type='fact')
    if scenarios is not None and len(scenarios) > 0:
        top_goals = []
        for scenario in scenarios:
            top_goals.append(scenario_to_goal_model(scenario))
        root_goal.children = top_goals

    print(str(root_goal))
    return root_goal


# 合并场景
def is_similar(c, core, threshold=0.6):
    if cal_sim(c, core, tokenizer, model) > threshold:
        return True
    else:
        return False


def merge_scenarios(feature: Feature):
    user_story = feature.user_story
    scenarios = user_story.scenarios
    if len(scenarios) > 1:
        core_scenarios = {}
        for scenario in scenarios:
            core, new_sentence = get_core(scenario.value)
            flag = False
            for c in core_scenarios.keys():
                if is_similar(c, new_sentence):
                    core_scenarios[c].append(scenario)
                    flag = True
                    break
            if not flag:
                core_scenarios[new_sentence] = [scenario]
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


def merge_goals_by_so_that(root: Goal):
    goals = root.children
    if len(goals) > 2:
        new_goals = []
        goals_by_so_that = {}
        for g in goals:
            if g.so_that is not None and g.so_that != '':
                flag = False
                g_so_that_core = get_core(g.so_that)[1]
                for so_that in goals_by_so_that.keys():
                    if is_similar(so_that, g_so_that_core, 0.8):
                        print(f"合并{so_that}和{g.so_that}")
                        goals_by_so_that[so_that].append(g)
                        flag = True
                        break
                if not flag:
                    goals_by_so_that[g_so_that_core] = [g]
            else:
                new_goals.append(g)
        for so_that in goals_by_so_that.keys():
            if len(goals_by_so_that[so_that]) <= 1:
                new_goals.extend(goals_by_so_that[so_that])
            else:
                new_goal = Goal(value=so_that, type='goal', so_that=so_that)
                for goal in goals_by_so_that[so_that]:
                    new_goal.add_child(goal)
                new_goals.append(new_goal)
        root.children = new_goals


def merge_goals_by_verb_and_obj(root: Goal):
    if root.children is not None and len(root.children) >= 2:
        for child in root.children:
            merge_goals_by_verb_and_obj(child)
        new_children = []
        verb_obj = {}
        for child in root.children:
            core, new_sentence = get_core(child.value)
            flag = False
            for c in verb_obj.keys():
                if is_similar(c, new_sentence):
                    verb_obj[c].append(child)
                    flag = True
                    break
            if not flag:
                verb_obj[new_sentence] = [child]
        for c in verb_obj.keys():
            if len(verb_obj[c]) <= 1:
                new_children.extend(verb_obj[c])
            else:
                new_goal = Goal(value=verb_obj[c][0].value, type='goal')
                for goal in verb_obj[c]:
                    new_goal.add_child(goal)
                new_children.append(new_goal)
        root.children = new_children
    return root


def merge_goals_by_verb(root: Goal):
    if root.children is not None and len(root.children) >= 2:
        for child in root.children:
            merge_goals_by_verb(child)
        new_children = []
        verb_dict = {}
        for child in root.children:
            core, _ = get_core(child.value)
            verb = ''
            subj = ''
            for token in core[0]:
                subj += token.text + ' '
            subj = subj.strip()
            for token in core[1]:
                verb += token.text + ' '
            verb = verb.strip()
            flag = False
            for c in verb_dict.keys():
                if is_similar(c, verb):
                    verb_dict[c].append([child, subj])
                    flag = True
                    break
            if not flag:
                verb_dict[verb] = [[child, subj]]
        for c in verb_dict.keys():
            if len(verb_dict[c]) <= 1:
                for goal in verb_dict[c]:
                    new_children.append(goal[0])
            else:
                new_goal = Goal(value=f"{verb_dict[c][0][1]} {c} something", type='goal')
                for goal in verb_dict[c]:
                    new_goal.add_child(goal[0])
                new_children.append(new_goal)
        root.children = new_children
        for child in root.children:
            child.father = root
    return root


def get_core(sentence):
    doc = nlp(sentence)
    passive = False
    for token in doc:
        print(token.text, token.dep_, token.head.text)
        if 'pass' in token.dep_:
            passive = True
    if not passive:
        sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
        verb_toks = [tok for tok in doc if (tok.pos_ == "VERB")]
        dobj_toks = [tok for tok in doc if (tok.dep_ == "dobj")]
        iobj_toks = [tok for tok in doc if (tok.dep_ == "dative")]
        if sub_toks:
            subject = [tok for tok in sub_toks[0].subtree]
        else:
            subject = []
        if verb_toks:
            verb_object = [tok for tok in verb_toks[0].subtree if tok.pos_ == "VERB"]
        else:
            verb_object = []
        # verb = verb_toks[0].text
        if dobj_toks:
            direct_object = [tok for tok in dobj_toks[0].subtree]
        else:
            direct_object = []
        if iobj_toks:
            indirect_object = [tok for tok in iobj_toks[0].subtree]
        else:
            indirect_object = []

        # print("Subject:", subject)
        # print("Verb:", verb_object)
        # print("Direct Object:", direct_object)
        # print("Indirect Object:", indirect_object)
        if len(verb_object) >= 2:
            temp = []
            auxs = [tok for tok in doc if tok.dep_ == "aux"]
            for v in verb_object:
                for aux in auxs:
                    if v == aux.head:
                        temp.append(aux)
                temp.append(v)
            verb_object = temp
        core = [subject, verb_object, direct_object, indirect_object]
    else:
        verb_toks = [tok for tok in doc if (tok.pos_ == "VERB")]
        if verb_toks:
            verb_object = [tok for tok in verb_toks[0].subtree if tok.pos_ == "VERB"]
        else:
            verb_object = []
        if len(verb_object) >= 2:
            temp = []
            auxs = [tok for tok in doc if tok.dep_ == "aux"]
            for v in verb_object:
                for aux in auxs:
                    if v == aux.head:
                        temp.append(aux)
                temp.append(v)
            verb_object = temp
        dobj_toks = [tok for tok in doc if (tok.dep_ == "nsubjpass")]
        # 第一个字母小写
        if dobj_toks:
            direct_object = [tok for tok in dobj_toks[0].subtree]
        else:
            direct_object = []
        # 输出动词原形
        verb = [tok for tok in doc if (tok.pos_ == "VERB")][0].lemma_
        sub_toks = [tok for tok in doc if (tok.dep_ == "pobj")]
        # 第一个字母大写
        if sub_toks:
            subject = [tok for tok in sub_toks[0].subtree]
        else:
            subject = []
        # print("Subject:", subject)
        # print("Verb:", verb)
        # print("Direct Object:", direct_object)
        core = [subject, verb_object, direct_object, None]
    new_sentence = ''
    for c in core:
        if c is not None:
            for t in c:
                new_sentence = new_sentence + t.lemma_ + ' '
    new_sentence = new_sentence.strip()
    return core, new_sentence


# 对根节点进行剪枝，如果root的子节点只有一个，那么就把root的子节点变成root的父节点的子节点
def prune(root: Goal):
    if root is not None and root.father is not None and len(root.children) == 1:
        root.children = root.children[0].children
        for child in root.children:
            child.father = root
    else:
        for child in root.children:
            prune(child)


def remove_modifiers(sentence: str):
    doc = nlp(sentence)
    new_sentence = []
    for token in doc:
        print(token.text, token.dep_, token.head.text, token.head.pos_)
        if token.dep_ not in ["amod", "advmod", "npadvmod"]:
            new_sentence.append(token.text)
    return " ".join(new_sentence)


def analyze_sentence(sentence):
    doc = nlp(sentence)
    passive = False
    for token in doc:
        print(token.text, token.dep_, token.head.text)
        if 'pass' in token.dep_:
            passive = True
    if not passive:
        sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
        verb_phrase = []
        for token in doc:
            if token.pos_ == "VERB":
                verb_phrase.append(token.text)
            elif token.dep_ == "prt" and token.head.pos_ == "VERB":
                verb_phrase.append(token.text)
        dobj_toks = [tok for tok in doc if (tok.dep_ == "dobj")]
        iobj_toks = [tok for tok in doc if (tok.dep_ == "dative")]
        if sub_toks:
            subject = [tok for tok in sub_toks[0].subtree]
        else:
            subject = []

        if dobj_toks:
            direct_object = [tok for tok in dobj_toks[0].subtree]
        else:
            direct_object = []
        if iobj_toks:
            indirect_object = [tok for tok in iobj_toks[0].subtree]
        else:
            indirect_object = []

        print("Subject:", subject)
        print("Verb:", verb_phrase)
        print("Direct Object:", direct_object)
        print("Indirect Object:", indirect_object)
        new_sentence = [subject, verb_phrase, direct_object, indirect_object]
    else:
        dobj_toks = [tok for tok in doc if (tok.dep_ == "nsubjpass")]
        # 第一个字母小写
        if dobj_toks:
            direct_object = [tok for tok in dobj_toks[0].subtree]
        else:
            direct_object = []
        # 输出动词原形
        # verb = [tok for tok in doc if (tok.pos_ == "VERB")][0].lemma_
        verb_phrase = []
        for token in doc:
            if token.pos_ == "VERB":
                verb_phrase.append(token.lemma_)
            elif token.dep_ == "prt" and token.head.pos_ == "VERB":
                verb_phrase.append(token.text)
        sub_toks = [tok for tok in doc if (tok.dep_ == "pobj")]
        # 第一个字母大写
        if sub_toks:
            subject = [tok for tok in sub_toks[0].subtree]
        else:
            subject = []
        print("Subject:", subject)
        print("Verb:", verb_phrase)
        print("Direct Object:", direct_object)
        new_sentence = [subject, verb_phrase, direct_object, []]
    return new_sentence


def dfs(root: Goal):
    if root is not None:
        if root.value == 'scenario_value':
            root.value = summarize_goals(root.children, tokenizer2, model2)
        for child in root.children:
            dfs(child)


def bfs(root: Goal):
    queue = [root]
    cnt = 1
    while queue:
        node = queue.pop(0)
        if node.context is not None and node.context.value == 'temp':
            node.context.value = 'Context' + str(cnt)
            cnt += 1
        for child in node.children:
            queue.append(child)


def extract_state_context(text):
    doc = nlp(text)
    result = []
    for token in doc:
        print(token.text, token.pos_, token.dep_, token.head.text, token.head.pos_, )
        if (token.dep_ == "ROOT" or token.pos_ == 'VERB') and token.lemma_ == "be":
            subject = [w for w in token.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                for w in token.children:
                    if w.dep_ == "acomp":
                        result.append((subject.text, w.text))
        elif token.dep_ == "ROOT" or token.pos_ == 'VERB':
            subject = [w for w in token.lefts if w.dep_ == "nsubj"]
            for w in token.children:
                if w.dep_ == "acomp" and w.pos_ == "ADJ":
                    result.append((subject[0].text, w.text))
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            result.append((token.head.text, token.text))
    return result


def similarity(s1, s2):
    result = cal_sim(s1, s2, tokenizer, model)
    print(f"{s1} and {s2} 的相似度: {result}")
    return result


def antonymy(s1, s2):
    antonyms = []
    for syn in wordnet.synsets(s1):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    similarities = [similarity(antonym, s2) for antonym in antonyms]
    if similarities:
        print(f"{s1} and {s2} 的反义度: {max(similarities)}")
        return max(similarities)
    else:
        return 0


def is_conflict(s1, s2, theshold):
    sim = similarity(s1, s2)
    ant = antonymy(s1, s2)
    if ant > theshold and ant >= sim:
        return True
    else:
        return False


# tokenizer2 = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
# model2 = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")

    # 计时
    start = time.time()
    feature_dir = './user_story_feature/BehatMage-develop'
    features_by_actor = {}
    goals_by_actor = {}
    for file in os.listdir(feature_dir):
        if file.endswith('.feature'):
            feature = parse_feature_file(os.path.join(feature_dir, file))
            if feature is not None:
                if feature.user_story.as_a not in features_by_actor.keys():
                    features_by_actor[feature.user_story.as_a] = []
                features_by_actor[feature.user_story.as_a].append(feature)

    for actor in features_by_actor.keys():
        goal = Goal(value=actor, type='goal')
        for feature in features_by_actor[actor]:
            if actor not in goals_by_actor.keys():
                goals_by_actor[actor] = []
            goals_by_actor[actor].append(feature_to_goal_model(feature))
        # goals_by_actor[actor] = merge_goals_by_so_that(goals_by_actor[actor])
        for g in goals_by_actor[actor]:
            goal.add_child(g)
        # goal = merge_goals_by_verb_and_obj(goal)
        # dfs(goal)
        bfs(goal)
        merge_goals_by_so_that(root=goal)
        goal = merge_goals_by_verb(goal)
        prune(goal)
        dot = Digraph(comment='Goal Model')
        draw_goal_model(goal, dot)
        dot.render('./result/goal_model_' + actor + '.png', view=True)
    end = time.time()
    print("time: ", end - start)
