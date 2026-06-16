from datetime import datetime


class IvetteContext:

    def __init__(self):

        self.mode = "Home"
        self.active_set = None
        self.active_run = None
        self.extra = {}


context = IvetteContext()



def render_header():

    print("\n" + "=" * 60)

    print(
        "IVETTE"
    )

    print(
        f"Mode: {context.mode}"
    )


    if context.active_set:

        print(
            f"Set: {context.active_set}"
        )


    if context.active_run:

        print(
            f"Run: {context.active_run}"
        )


    for key, value in context.extra.items():

        print(
            f"{key}: {value}"
        )


    print(
        "=" * 60
    )