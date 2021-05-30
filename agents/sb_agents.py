import os
import pickle
from utils.eval import eval_policy, evaluate
from datetime import datetime

class SBAgent(object):
    def __init__(self, model, train_env,  name:str = 'PPO'):
        self.model = model
        self.train_env = train_env
        self.pretrained_path = 'pretrained'
        self.log_path = os.path.join(self.pretrained_path, datetime.now().strftime("%Y-%m-%d %H: %M") + " agent_" + name)
        self._path_check()
        
    def train(self, total_timesteps:int = 300000):
        return self.model.learn(total_timesteps = total_timesteps)
        
    def test(self, n_episodes, deterministic = True):
        return evaluate(self.model, n_episodes, deterministic)
    
    def eval(self, test_env, n_episodes):
        return eval_policy(self.model, test_env, n_episodes)
    
    def save(self, path):
        self._path_check()
        self.model.save(os.path.join(self.log_path, path))
        
    def load(self, path):
        return self.model.load(os.path.join(self.log_path, path))
        
        
    def save_dict(self, path):
        _path = os.path.join(self.log_path, path)
        self._save_obj(self.model.__dict__, _path)
        
    def load_dict(self, path):
        _path = os.path.join(self.log_path, path)
        return self._load_obj(_path)
               
        
    def _path_check(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def _save_obj(self, obj:object, path:str = None) -> None:
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def _load_obj(self, path:str = None) -> object:
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)