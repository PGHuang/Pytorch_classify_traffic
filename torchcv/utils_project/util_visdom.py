import numbers
import shutil
import os.path as path
import numpy as np
import visdom


class Visualizer(object):
    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.last_x = {}
        self.wins = {}
        self.env = env
        # close all win when init
        self.close_win()

    def text(self, win_name, content):
        """
            draw text
            win_name (str): window name
            content (str or dict):  content of text
        """
        info = win_name + ':<br>'

        if isinstance(content, str):
            info = info + content
        elif isinstance(content, dict):
            for k, v in content.items():
                info += str(k) + ' = ' + str(v) + '<br>'
        else:
            raise Exception('content must be string or dict like object')

        self.vis.text(info, win=win_name, opts=dict(title=win_name))

    def line(self, win_name, value, x=None, step=1, **kwargs):
        """
            draw lines
            name (str): window name
            value (number or dict): y coord, if value is dict then draw multiple
                                    lines and legend will keys
            x (None or number): x coord, if x=None then x will be number start
                                from 0 increase with step
        """
        if x is None:
            x = self.last_x.get(win_name, -1) + step

        opts = dict(title=win_name)
        if isinstance(value, numbers.Number):
            X = np.array([x])
            Y = np.array([value])
        elif isinstance(value, dict):
            legend = []
            X, Y = [], []
            for k, v in sorted(value.items()):
                X.append(x)
                Y.append(v)
                legend.append(k)
            opts['legend'] = legend
            X = np.array([X])
            Y = np.array([Y])
        else:
            raise Exception('value must be a number or a dict like object')
        if win_name not in self.wins:
            # self.wins[name] = self.vis.line(X=X, Y=Y, opts=opts, name=name, **kwargs)
            self.wins[win_name] = self.vis.line(X=X, Y=Y, opts=opts, **kwargs)
        else:
            self.vis.line(X=X, Y=Y, win=self.wins[win_name], opts=opts, update='append', **kwargs)
        self.last_x[win_name] = x

    def image(self, win_name, imgs, **kwargs):
        """
            draw images
            imgs:(np.ndarray or list of np.ndarray): axis order (B, C, H, W)
        """
        self.vis.images(imgs, win=win_name, opts=dict(title=win_name), **kwargs)

    def close_win(self, win_name=None):
        """
            close window
            name (str): name of window, None means close all windows
        """
        self.vis.close(win=win_name, env=self.env)

    def delete(self):
        """
        delete environment from host
        """
        self.vis.delete_env(self.env)

    def save(self, folder=None):
        """
        save environment to a json file
        folder (str):   destination folder of json file
        """
        self.vis.save(envs=[self.env])

        if folder is not None:
            home = path.expanduser('~')
            vis_path = path.join(home, '.visdom', self.env + '.json')
            shutil.copy2(vis_path, folder)
