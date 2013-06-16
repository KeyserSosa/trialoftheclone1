import re
import os

from playgame import Transition, Node, Act, draw_all, LOSE, NextNode, LosingNode, WinningNode

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
                act.add(pageno, topage, transition)
            else:
                if "YOU LOSE" in line or "THE END" in line:
                    act.make_transitional(pageno, LosingNode)
                elif "YOU WIN" in line:
                    act.make_transitional(pageno, WinningNode)
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
                    elif line.startswith("go to act "):
                        act.make_transitional(pageno, NextNode)

        if line.strip():
            last_line = line


def test(path = "episode2"):
    acts = {}
    for i in range(5):
        n = i + 1
        print "processing Act %s..." % n
        f = os.path.join(path, "act%d.txt" % n)
        acts[n] = act = Act(str(n))
        with open(f) as handle:
            process_act(act, handle)
        #draw_all([act])
        act.draw()
        print act.lose_map
    draw_all([a for n, a in sorted(acts.iteritems())])

if __name__ == "__main__":
    test()
