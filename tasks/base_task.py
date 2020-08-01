import sys


class MetaTask(type):
    def __new__(cls, cls_name, cls_base, cls_dict):
        if cls_dict.get("group", False):
            task_map = dict(
                [(k.split("_", 1)[1], v)
                 for k, v in cls_dict.items()
                    if k.startswith("task_") and callable(v)])
            cls_dict["task_map"] = task_map

            def call(instance, *args, **kwargs):
                main_task = kwargs.pop("main_task", False)
                if main_task:
                    return instance.call(*args, **kwargs)
                else:
                    return task_map.get(args[0])(instance, *args[1:], **kwargs)
            cls_dict["run"] = call

        new_cls = super().__new__(cls, cls_name, cls_base, cls_dict)
        return new_cls


class BaseTask(metaclass=MetaTask):

    @classmethod
    def help(cls, *args, **kwargs):
        print("list or start")

    @classmethod
    def run_tasks(cls, *args, **kwargs):
        argv = sys.argv
        if len(argv) < 2:
            return cls.help(cls)
        else:
            subcmd = argv[1]
            args = argv[2:]
            if subcmd == "list":
                subfunc = cls.list
            else:
                subfunc = cls.start
            return subfunc(*args)

    @classmethod
    def list(cls, *args, **kwargs):
        subclses = cls.__subclasses__()
        for subcls in subclses:
            print(subcls.task_name, " - ", subcls.__doc__)
            if "group" in subcls.__dict__ and subcls.group:
                for task_name, task_func in subcls.task_map.items():
                    print((
                        f"{subcls.task_name}.{task_name} "
                        f"- {task_func.__doc__}"
                    ))

    @classmethod
    def start(cls, *args, **kwargs):
        task_name = args[0]
        task_args = args[1:]
        subtask = None
        if "." in task_name:
            task_name, subtask = tuple(task_name.split(".", 1))
        task_cls = [subcls for subcls in cls.__subclasses__(
        ) if subcls.task_name == task_name][0]
        if subtask:
            return task_cls()(*([subtask] + list(task_args)), **kwargs)
        else:
            kwargs["main_task"] = True
            return task_cls()(*task_args, **kwargs)

    def call(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
