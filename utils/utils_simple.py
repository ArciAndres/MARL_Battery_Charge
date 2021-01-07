def get_command(params):
   # Returns the parameters put in the command line syntax
   # Consider that the bool parameters are defined by default in the argparse function
   # And the value inserted in the dictionary represents nothing, but a type indicator
    command = ""
    for k, v in params.items():
        command += '--%s ' %k 
        if type(v) != bool: 
            command += '%s ' % v
    print(command)
    return command