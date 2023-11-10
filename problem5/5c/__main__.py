import gymnasium as gym

from .learn_env import EnvModel

def parse_args(preset, render_mode, out, episodes):
    import sys
    args = sys.argv[1:]
    pos_args = []
    kw_args = {}
    kw = None
    current_args = []
    for arg in args:
        if arg.startswith("--"):
            if kw is not None:
                kw_args[kw] = current_args
                current_args = []
            else:
                pos_args = current_args
            kw = arg
        else:
            current_args.append(arg)
    if kw is not None:
        kw_args[kw] = current_args
    else:
        pos_args = current_args

    if len(pos_args) > 0:
        preset = pos_args[0]
    if "--render" in kw_args:
        if len(kw_args["--render"]) == 0:
            render_mode = "human"
        else:
            render_mode = kw_args["--render"][0]
        del kw_args["--render"]
    if "--preset" in kw_args and len(kw_args["--preset"]) > 0:
        preset = kw_args["--preset"][0]
        del kw_args["--preset"]
    if "--out" in kw_args and len(kw_args["--out"]) > 0:
        out = kw_args["--out"][0]
        del kw_args["--out"]
    if "--episodes" in kw_args and len(kw_args["--episodes"]) > 0:
        episodes = int(kw_args["--episodes"][0])
        del kw_args["--episodes"]

    kwargs = {}
    for key in kw_args:
        if len(kw_args[key]) == 0:
            if key.startswith("--not_"):
                kwargs[key[6:]] = False
            else:
                kwargs[key[2:]] = True
        elif len(kw_args[key]) == 1:
            if kw_args[key][0] == "None":
                kwargs[key[2:]] = None
            elif kw_args[key][0] == "True":
                kwargs[key[2:]] = True
            elif kw_args[key][0] == "False":
                kwargs[key[2:]] = False
            else:
                kwargs[key[2:]] = kw_args[key][0]
        else:
            kwargs[key[2:]] = kw_args[key]

    return preset, render_mode, out, episodes, kwargs



if __name__ == '__main__':
    preset, render_mode, out, episodes, kwargs = parse_args("CartPole-v1", None, "model.pth", 1000)

    env = gym.make(preset, render_mode=render_mode, **kwargs)
    model = EnvModel(env)
    model.train(num_episodes=episodes)
    model.save(out)

