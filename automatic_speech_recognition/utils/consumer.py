import os
import logging
import subprocess
import time
logger = logging.getLogger('asr.consumer')


def next_command(file_name):
    """ Get the first file line - truncate to the current pointer position."""
    with open(file_name, "r+") as f:
        line = f.readline()
        remaining_lines = f.readlines()
        f.seek(0)
        f.writelines(remaining_lines)
        f.truncate()
    return line


def execute(python: str, command: str):
    """ Execute a python script and waits for it to finish (security issue). """
    console, = logger.handlers      # Consumer has only one handler because it is often run via nohup
    subprocess.run([python, command], stderr=console.stream, cwd=os.getcwd(), check=True)


def run_consumer(queue: str, python: str):
    """ Run program in the infinite loop (a daemon process). """
    while True:
        try:
            command = next_command(queue)
            if command:
                logger.info(f'Consumer attempt to run the command: {command}')
                execute(python, command)
                logger.info(f'Successful run: {command}')
            else:
                time.sleep(10)

        except subprocess.CalledProcessError:
            logger.error(f'Consumer can not execute the command')
