
def alignCharacter(multiline_string: str, character_to_align: str) -> str:
    lines = multiline_string.split("\n")
    character_locations = [line.find(character_to_align) for line in lines]
    move_character_to = max(character_locations)
    for i, (line, loc) in enumerate(zip(lines, character_locations)):
        if loc < 0:
            continue

        lines[i] = line[:loc] + " "*(move_character_to - loc) + line[loc:]

    return "\n".join(lines)


def indent(level: int, multiline_string: str, tab: str=" "*4) -> str:
    lines = multiline_string.split("\n")
    lines = [tab*level + line for line in lines]
    return "\n".join(lines)
