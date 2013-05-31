import re
import os

from playgame import Transition, Node, Act, draw_all, LOSE

re_transition = re.compile(r"^(.*?) *\[\[(\d+)\]\]")


def process_act(act, handle):
    pageno = None
    last_line = None

    for line in handle:
        # new page!
        if line.startswith(":: "):
            pageno = line.split()[1]
        else:
            m = re_transition.search(line)
            if m:
                transition, topage = m.groups()
                if not transition:
                    transition = last_line.strip()
                print pageno, topage, transition
                act.add(pageno, topage, transition)
            else:
                if "YOU LOSE" in line:
                    act.add(pageno, LOSE, "you lose")
                else:
                    line = line.lower()
                    if line.startswith("battle"):
                        line = line.replace(" and ", "; ")
                        line = line.replace("fighting ", "fighting: ")
                        # fill in some starter values:
                        line = line.replace("fighting", "fighting =1; hp =1")
                        line = line.replace("wits", "wits =1")
                        line = line.replace("charisma", "charisma =1")
                        # Flag battle sequences where applicable
                        act[pageno].set_kind(line)

        if line.strip():
            last_line = line

    # at this point, we've got a bunch of pages and a bunch of transitions.
    # Turn these into an Act


def test(path = "episode2"):
    acts = {}
    for i in range(5):
        n = i + 1
        print "processing Act %s..." % n
        f = os.path.join(path, "act%d.txt" % n)
        acts[n] = act = Act(str(n))
        with open(f) as handle:
            process_act(act, handle)
        print act.lose_map


if __name__ == "__main__":
    test()
