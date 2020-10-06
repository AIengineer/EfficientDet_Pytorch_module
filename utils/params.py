import json
import re

import yaml


class YamlParams:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

    def to_dict(self):
        return self.params


class JsonConfig:
    def __init__(self, config_file):
        self.params = json.load(open(config_file))
        self.params = self.to_snake_case()

    def __getattr__(self, item):
        return self.params.get(item, None)

    def to_snake_case(self):
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        return {pattern.sub('_', k).lower(): v for k, v in self.params.items()}

    def to_pascal_case(self):
        return {''.join([k.title() for k in key.split('_')]): value for key, value in self.params.items()}

    def to_dict(self):
        return self.params
