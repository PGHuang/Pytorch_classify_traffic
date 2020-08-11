import json


class ConfigBase(object):
    # 把类中的参数转换成Dict对象
    def to_dict(self):
        d = {}
        for key, value in vars(self.__class__).items():
            if not key.startswith('__') and key != 'to_dict' and key != 'to_json':
                if callable(value):
                    d[key] = value.__name__
                else:
                    d[key] = value
        return d

    # 把类中的参数保存成JSON文件
    def to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent="    ")

    def to_show(self):
        dict_param = self.to_dict()
        print("*****" * 6, "  Parameters  ", "*****" * 6)
        print(json.dumps(dict_param, indent="    "))
        print("*****" * 15, "\n")

    def to_text(self):
        dict_param = self.to_dict()
        info = ""
        for k, v in dict_param.items():
            info += str(k) + ' = ' + str(v) + '<br>'
        return info
