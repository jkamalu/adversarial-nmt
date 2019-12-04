
# some useful decorators
def time_this(original_function):
    def new_function(*args,**kwargs):
        print("starting timer")
        import datetime
        before = datetime.datetime.now()
        x = original_function(*args,**kwargs)
        after = datetime.datetime.now()
        print("Elapsed Time = {0}".format(after-before))
        return x
    return new_function
