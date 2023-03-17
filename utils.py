import os


def scan_feature(dir_path, save_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.feature'):
                with open(os.path.join(root, file), 'r',encoding='utf8') as f:
                    content = f.read()
                with open(os.path.join(save_path, file,), 'w',encoding='utf8') as f:
                    f.write(content)


if __name__ == '__main__':
    for file in os.listdir('dataset'):
        # 如果 os.path.join('user_story_feature', file) 不存在，创建
        if not os.path.exists(os.path.join('user_story_feature', file)):
            os.mkdir(os.path.join('user_story_feature', file))

        scan_feature(os.path.join('./dataset', file), os.path.join('user_story_feature', file))
