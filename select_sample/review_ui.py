import webbrowser


class Back(Exception):
    """Undo the last answer given in this session."""


class Skip(Exception):
    """Leave the current thread for a later session."""


class Quit(Exception):
    """Stop the review."""


class OpenThread(Exception):
    """Open the current thread in the browser."""


COMMANDS = {
    ":back": Back, ":b": Back,
    ":skip": Skip, ":s": Skip,
    ":open": OpenThread, ":o": OpenThread,
    ":quit": Quit, ":q": Quit,
}

COMMANDS_HELP = (
    ":back undo last answer | :skip leave thread for later | "
    ":open thread in browser | :quit exit"
)


def read(prompt):
    text = input(prompt).strip()
    command = COMMANDS.get(text.lower())
    if command:
        raise command
    return text


def ask(field):
    if field.options is None:
        answer = ""
        while not answer:
            answer = read(f"{field.prompt}\n> ")
        return answer.upper() if field.upper else answer

    keys = "/".join(field.options)
    answer = ""
    while answer not in field.options:
        answer = read(f"{field.prompt} ({keys}): ").lower()
    return field.options[answer]


def thread_url(row):
    return f"https://lore.kernel.org/all/{row['thread_id'].strip('<>')}/"


def ask_field(row, field):
    """Asks one field. `:open` opens the thread in the browser and repeats
    the question."""
    while True:
        try:
            return ask(field)
        except OpenThread:
            url = thread_url(row)
            if not webbrowser.open(url):
                print(f"Could not open a browser. URL: {url}")