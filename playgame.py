#!/usr/bin/env python

"""
Remaining todos:
 * rays generate status on enemy
 * missing some inventory items
"""

import math
import re
import csv
import logging
from StringIO import StringIO

if __name__ != "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
else:
    logger = logging
    logger.basicConfig(level = logging.INFO)

import os
import pydot
from random import choice, shuffle


MINROLL = 0
MAXROLL = 3


def roll():
    return choice(range(MINROLL, MAXROLL + 1))


def _hex(x):
    x = max(0, min(x, 255))
    _, x = hex(int(x)).split("x")
    return ("0%s" if len(x) == 1 else "%s") % x


def hexcolor(r, g, b):
    return "#" + "".join(_hex(x) for x in (r, g, b))


def spectrum(fr, to, frac):
    return hexcolor(*[x * (1 - frac) + y * frac for x, y in zip(fr, to)])


def mean(array):
    return float(sum(array)) / (len(array) or 1)


def sigma(array):
    return math.sqrt(max(sum(x ** 2 for x in array) / (len(array) or 1) -
                         mean(array) ** 2, 0))


class Player():
    fight_roll_multiplier = 1.
    wits_roll_multiplier = 1.

    def __init__(self, hp = 1, wits = 1, charisma = 1, fighting = 1):
        self.hp = hp
        self.max_hp = None

        self.wits = wits
        self.charisma = charisma
        self.fighting = fighting

        self.inventory = {}
        self.specialize = {}

    def is_dead(self):
        return self.hp <= 0

    def use_medipack(self):
        if "medi-pack" in self.inventory:
            self.hp = self.max_hp
            del self.inventory['medi-pack']
            return True
        return False

    def __repr__(self):
        return "<%s {%s} (HP=%d) f=%d/w=%d/c=%d>" % (
            self.__class__.__name__, self.specialize.keys(),
            self.hp, self.fighting,
            self.wits, self.charisma)

    def attack(self, other):
        damage = self.fighting + int(roll() * self.fight_roll_multiplier)
        logging.debug("%s dealt %s %d damage", self, other, damage)
        other.hp -= damage
        other.hp = max(0, other.hp)


class Opponent(Player):
    pass


class Transform(object):
    """
    Base class for transformations, i.e., adding specializations or changing
    stats of the player.

    These classes exist primarily to make parsing easier.  Apply and inverse are
    used on the actual player.
    """

    # list of words that apply to this sort of transformation
    keywords = ()

    def __init__(self, what):
        self.what = what

    def apply(self, person):
        raise NotImplementedError(self)

    def inverse(self):
        raise NotImplementedError(self)

    @classmethod
    def parse(cls, s):
        _, _, x = s.partition(" ")
        return cls(x.lower().strip())

    @classmethod
    def couldbe(cls, s):
        """
        Used to help parsing out
        """
        s = s.strip().lower()
        for k in cls.keywords:
            if s.startswith(k):
                return True

    def __repr__(self):
        name = self.__class__.__name__.lower()
        if name.lower().endswith("transform"):
            name = name[:-9]
        #return "[%s:%s]" % (name, self._what_repr())
        return self._what_repr()

    def _what_repr(self):
        return self.what

    def export(self, stream):
        raise NotImplementedError


class QualityTransform(Transform):
    """
    deals with character trait transformations
    """
    keywords = ("fighting", "wits", "charisma", "hp")

    def __init__(self, what, kind, val):
        self.what = what
        self.kind = kind
        self.val = val
        self.prev_val = None

    def inverse(self):
        if self.kind != "=":
            return self.__class__(self.what,
                                  "-" if self.kind == "+" else "+",
                                  self.val)
        else:
            return self.__class__(self.what, "=",
                                  self.prev_val)

    def apply(self, person):
        v = getattr(person, self.what)
        self.prev_val = v
        if self.kind == "=":
            v = self.val
        elif self.kind == "+":
            v += self.val
        elif self.kind == "-":
            v -= self.val
            v = max(v, 0)

        setattr(person, self.what, v)

    def _what_repr(self):
        return "%s %s%s" % (self.what, self.kind, self.val)

    @classmethod
    def parse(cls, s):
        if "=" in s:
            what, kind, val = s.partition('=')
        elif "+" in s:
            what, kind, val = s.partition('+')
        elif "-" in s:
            what, kind, val = s.partition('-')
        val = val.strip()
        # TODO: temporary
        if val.lower() == 'x':
            val = 1
        else:
            val = int(val)
        return cls(what.strip(), kind.strip(), val)

    def export(self, stream):
        stream.write("%s %s%s" % (self.what, self.kind, self.val))


class BecomeTransform(Transform):
    """
    deals with become and specialize class behavior
    """
    keywords = ("become", "specialize")

    classes = ["medic", "engineer", "fighter"]

    def apply(self, player):
        i = Specialization.get(self.what)
        if self.what in self.classes:
            for k in self.classes:
                if k in player.specialize:
                    player.specialize[k].remove(player)
                    del player.specialize[k]
        player.specialize[self.what] = i
        i.apply(player)


class GainTransform(Transform):
    keywords = ("gain", )

    rays = ["alpha ray", "beta ray", "mind-bending ray", "ultra-blaster"]
    plasmasters = ["goodsider plasmaster",
                   "basic plasmaster"]

    def apply(self, player):
        i = Item.get(self.what)
        if self.what in self.rays:
            for k in self.rays:
                if k in player.inventory:
                    player.inventory[k].remove(player)
                    del player.inventory[k]
        elif self.what in self.plasmasters:
            if not (self.what == "basic plasmaster" and
                    "goodsider plasmaster" in player.inventory):
                for k in self.plasmasters:
                    if k in player.inventory:
                        player.inventory[k].remove(player)
                        del player.inventory[k]

        if not (self.what == "basic plasmaster" and
                "goodsider plasmaster" in player.inventory):
            player.inventory[self.what] = i
        i.apply(player)

    def export(self, stream):
        stream.write("gain %s" % self.what)


class ResetTransform(Transform):
    keywords = ("reset", "lose all")

    def apply(self, player):
        for item in player.inventory.values():
            item.remove(player)
        for item in player.specialize.values():
            item.remove(player)

    def export(self, stream):
        stream.write("lose all")

class LoseTransform(Transform):
    keywords = ("lose", "otherwise", "you fail", "you lose")

    def __repr__(self):
        return "[lose]"

    def apply(self, player):
        pass


class HaveTransform(Transform):
    """
    Transformation used for branching: checks if the player has
    The item specified.
    """

    keywords = ("you've",)

    def __init__(self, what):
        self.optional = False
        if " wish " in what:
            what, _, _ = what.partition(' wish ')
            self.optional = True
        if " want " in what:
            what, _, _ = what.partition(' want ')
            self.optional = True
        Transform.__init__(self, what)

    def apply(self, player):
        return self.what in player.inventory

    def inverse(self):
        return self


class AreTransform(Transform):
    keywords = ("you're",)

    def __init__(self, what):
        self.optional = False
        if what.startswith("optionally"):
            _, _, what = what.partition(' ')
            self.optional = True
        Transform.__init__(self, what)
        self.specialization = (what in BecomeTransform.classes)

    def inverse(self):
        return self

    def apply(self, player):
        if self.specialization and self.what not in player.specialize:
            return False
        return True


class Status(object):
    keyword = None

    seen = {}

    def __init__(self, transforms, name):
        self.name = name
        self.transforms = transforms
        self.rtransforms = []
        for t in transforms:
            self.rtransforms.append(t.inverse())
        self.seen[name] = self

    @classmethod
    def parse(cls, line):
        _, transforms, name = line.split(":")
        transforms = parse_rules(transforms)
        return cls(transforms, name.strip())

    @classmethod
    def couldbe(cls, s):
        s = s.strip()
        return s.startswith(cls.keyword + ":")

    def apply(self, player):
        for t in self.transforms:
            t.apply(player)

    def remove(self, player):
        for t in self.rtransforms:
            t.apply(player)

    @classmethod
    def get(cls, name):
        return cls.seen[name]

    def export(self, stream):
        stream.write("%s: " % self.keyword)
        for i, t in enumerate(self.transforms):
            t.export(stream)
            if i != len(self.transforms) - 1:
                stream.write("; ")
        stream.write(": %s\n" % self.name)

class Item(Status):
    keyword = "item"


class Specialization(Status):
    keyword = "specialization"


ignore = ("you wish", "you refuse", "you continue",
          "you say", "you do", "you surrender", "you try",
          "you decide", "you keep", "you throw", "you run", "you make",
          "you spin", "you head", "you flee", "you stand", "you use",
          "some shit", "you attempt", "you begin", "you bombard",
          "you challenge", "you confess", "you creep",
          "you defeat", "you duck", "you elect", "you enter",
          "you examine", "you follow", "you hack", "you build",
          "you hide", "you hop", "you ignite", "you laugh",
          "you leap", "you let", "you locate", "you look",
          "you must", "you nobly", "you note", "you plan",
          "you point", "you pretend", "you pull", "you punch", "you push",
          "you seize", "you shoot", "you shove", "you hold", "you karate",
          "you remain", "you report", "you slap", "you sportingly",
          "you stab", "you wuss", "you were", "you urge", "you train",
          "you touch", "you suck", "you stop", "you stay", "you survive",
          "you talk", "you eat", "you ignore", "you got", "you lay",
          "you promise", "you ride", "you spend", "you tell", "you wait",
          "you attack", "you hire", "you kill", "you insist", "you drink",
          "you join", "you fire", "you fight", "you go", "you choose",
          "you've ever", "you've not", "you succeed", "you take", "you think",
          "take controls", "you focus", "remove")
battle_re = re.compile("\(battling.*\)")
win_re = re.compile("^you win,? *")

dunno = {}


def parse_rules(s):
    res = []
    if "TK" in s:
        print " --- TK:", s
        return res
    s = battle_re.sub("", s)
    s = win_re.sub("", s)
    s = s.replace("you wish to be", "become")
    s = s.replace("you choose to be", "become")
    s = s.replace("you go into", "specialize")
    s = s.replace(" an ", " ")
    s = s.replace(" a ", " ")
    s = s.replace(" the ", " ")
    s = s.replace(",", ", ")
    s = s.replace(" and ", " ")
    s = s.replace("you are", "you're")
    s = s.replace("you have", "you've")
    s = s.replace("  ", " ")
    s = s.strip()
    s = s.strip(",")

    if not s or s == "not" or s == "do not":
        return res

    for i in ignore:
        if s.startswith(i):
            return res

    for part in re.split('[;,]', s):
        part = part.strip().lower()
        if part and part != "(always)":
            for transform in (QualityTransform, BecomeTransform,
                              GainTransform, ResetTransform,
                              LoseTransform, HaveTransform,
                              AreTransform):
                if transform.couldbe(part):
                    res.append(transform.parse(part))
                    break
            else:
                k = tuple(part.split()[:2])
                dunno.setdefault(k, []).append(part)
                logger.error("Parse error: %s'", part)
    return res


class Battle(object):
    seen = []

    def __init__(self, who, transforms, act):
        self.who = who
        self.transforms = transforms
        self.act = act

        self.fight = any(x.what == 'fighting' for x in transforms)
        self.wits = any(x.what == 'wits' for x in transforms)
        self.charisma = any(x.what == 'charisma' for x in transforms)

        # for keeping track of statistics of the battle
        self.nbattles = 0
        self.nwins = 0

        self.nrolls = []
        self.damage_dealt = []
        self.damage_absorbed = []

        self.initial_hp = []
        self.witvantage = []

        self.seen.append(self)

    def __repr__(self):
        kind = []
        if self.fight:
            kind.append("F")
        if self.wits:
            kind.append("W")
        if self.charisma:
            kind.append("C")
        return "Battle %s" % self.who

    @classmethod
    def parse(cls, label, act):
        _, label, who = label.lower().split(':')
        transforms = parse_rules(label)
        who = who.strip()
        return cls(who, transforms, act)

    @classmethod
    def couldbe(cls, label):
        return label.lower().startswith("battle")

    @classmethod
    def summarize(cls):
        def mean(l):
            return sum(l) / float(len(l))
        with open("/tmp/battles.txt", "w") as handle:
            handle.write("    WIN%    L     W  attack  absorb  hp0    rolls  who\n")
            handle.write("-" * 54 + "\n")
            act = None
            for b in cls.seen:
                if b.fight and b.nbattles:
                    if b.act != act:
                        act = b.act
                        handle.write("For Act %s:\n" % act)
                    handle.write("%8.2f%% %5d %5d %6.2f %6.2f %6.2f %6.3f %s\n" % \
                        ((100. * b.nwins) / b.nbattles,
                        (b.nbattles - b.nwins),
                         b.nwins,
                         mean(b.damage_dealt),
                         mean(b.damage_absorbed),
                         mean(b.initial_hp),
                         mean(b.nrolls),
                         b.who))

            handle.write("\n     WIN%    L     W  advantage   who\n")
            handle.write("-" * 50 + "\n")
            for b in cls.seen:
                if (b.wits or b.charisma) and b.nbattles:
                    if b.act != act:
                        act = b.act
                        handle.write("For Act %s:\n" % act)
                    handle.write("%8.2f%% %5d  %5d  %6.2f  %s\n" % \
                        ((100. * b.nwins) / b.nbattles,
                        (b.nbattles - b.nwins),
                         b.nwins,
                         mean(b.witvantage),
                         b.who))

    def apply(self, player, will_lose = False):
        self.nbattles += 1

        opponent = Opponent()
        for transform in self.transforms:
            transform.apply(opponent)

        # if it is a battle of wits, best out of 3
        if self.wits or self.charisma:
            randw = max(roll() for _ in range(3))
            randw += player.wits if self.wits else player.charisma
            opp = opponent.wits if self.wits else opponent.charisma
            self.witvantage.append(randw - opp)
            if randw < opp:
                return False

        # battle till the death if applicable!
        if self.fight:
            hp_player = player.hp
            self.initial_hp.append(hp_player)
            hp_opponent = opponent.hp
            attacker, defender = player, opponent
            rolls = 0
            while True:
                rolls += 1
                attacker.attack(defender)
                if defender.is_dead():
                    if not will_lose or not defender.use_medipack():
                        break
                # swap the roles
                attacker, defender = defender, attacker

            # store the damage dealt
            self.damage_dealt.append(hp_opponent - opponent.hp)
            self.damage_absorbed.append(hp_player - player.hp)
            self.nrolls.append(rolls)

            # if it is the player, he lost
            if defender is player:
                return False
        logging.debug("Successful battle: %r; wits = %s, fighting = %s",
                     player, self.wits, self.fight)

        self.nwins += 1
        return True

    def export(self, stream):
        stream.write("battle: ")
        for i, t in enumerate(self.transforms):
            t.export(stream)
            if i != len(self.transforms) - 1:
                stream.write("; ")
        stream.write(": %s" % self.who)


class Minigame(object):
    def __init__(self, name = None):
        self.name = name

    @classmethod
    def parse(cls, s):
        _, _, s = s.partition(":")
        s = s.strip()
        name, _, _ = s.partition(":")
        return Minigame(name)

    def apply(self, player, **kw):
        func = getattr(self, "do_" + self.name)
        return func(player)

    def do_statmash(self, player):
        for k in ("fighting", "wits", "charisma", "hp"):
            v = getattr(player, k)
            if roll() < 2:
                setattr(player, k, v - 3)
            else:
                setattr(player, k, v + 3)

    def do_roulette(self, player):
        return (roll() >= 2)

    def do_witrolling(self, player):
        wits = 0
        while True:
            r = roll()
            if r == 1:  # 1 <= r <= 4:
                wits += 2
            elif r == 2:  # 5 < r <= 8:
                wits -= 2
            else:
                break
        player.wits += wits
        if wits >= 10:
            return True
        return False

    @classmethod
    def couldbe(cls, label):
        return label.strip().lower().startswith("minigame")

    def export(self, stream):
        stream.write("minigme: %s" % self.name)

class Node(object):
    def __init__(self, label, act = -1):
        self.label = label
        self.game = None
        self.transforms = []
        self.oneoff = []
        self.traversed = 0
        self.act = act

    @classmethod
    def is_kind(cls, label):
        label = label.lower()
        return any(label.startswith(x)
                   for x in ("battle", "minigame", "this",
                             "wits", "fighting", "charisma",
                             "gain", "meet", "lose all"))

    @classmethod
    def is_transitional(cls, label):
        if label == "gameover":
            return LosingNode
        elif label == "win":
            return WinningNode
        elif label == "nextact":
            return NextNode

    def set_kind(self, label):
        if Battle.couldbe(label):
            self.game = Battle.parse(label, self.act)
            self.label += " " + repr(self.game)
        elif Minigame.couldbe(label):
            self.game = Minigame.parse(label)
        elif label.startswith("this"):
            _, _, rules = label.partition(":")
            rules = rules.strip()
            self.oneoff = parse_rules(rules)
        else:
            for t in (GainTransform, ResetTransform, QualityTransform):
                if t.couldbe(label):
                    self.transforms.append(t.parse(label))

    def apply(self, player, will_lose = False):
        self.traversed += 1
        res = None
        if self.oneoff:
            for rule in self.oneoff:
                if rule.apply(player) is False:
                    break
        for transform in self.transforms:
            transform.apply(player)
        if self.game is not None:
            res = self.game.apply(player, will_lose = will_lose)
        if self.oneoff:
            for rule in self.oneoff:
                if rule.inverse().apply(player) is False:
                    break
        return res

    def __repr__(self):
        return "(%s)" % self.label

    def __hash__(self):
        return hash(self.label)

    def draw(self, graph, n = 0):
        kw = {}
        if self.label.endswith('.1'):
            kw.update(dict(fillcolor = "green", style = "filled"))
        elif self.game:
            if getattr(self.game, "nbattles", 0):
                winpct = float(self.game.nwins) / self.game.nbattles
                color = spectrum((255, 0, 0), (0, 255, 0), winpct)
                if getattr(self.game, "fight"):
                    kw['shape'] = "rectangle"
            else:
                winpct = .5
                color = "#75B0CF"
                kw['shape'] = "rectangle"
            kw.update(dict(fillcolor = color, style = "filled"))
        if self.traversed < .002 * n:
            if not kw.get('fillcolor'):
                kw['fillcolor'] = "yellow"
            kw['style'] = 'filled'
        self._draw(graph, **kw)

    def _draw(self, graph, **kw):
        kw.setdefault("fontsize", 40)
        graph.add_node(pydot.Node(self.label, **kw))

    def number(self):
        return self.label.partition(".")[-1].partition(' ')[0]

    def export(self, stream):
        if self.game:
            stream.write("%s, " % self.number())
            self.game.export(stream)
        else:
            interesting = []
            for t in self.transforms:
                if isinstance(t, (GainTransform, ResetTransform)):
                    interesting.append(t)
            if interesting:
                stream.write("%s, " % self.number())
                for t in interesting:
                    t.export(stream)


class LosingNode(Node):
    def draw(self, graph, n = 0):
        self._draw(graph, fillcolor = "red",
                   style = "filled", shape = "octagon")

    def kind(self):
        return "gameover"


class WinningNode(Node):
    def draw(self, graph, n = 0):
        self._draw(graph, fillcolor = "green",
                   style = "filled", shape = "rectangle")

    def kind(self):
        return "win"


class NextNode(Node):
    def draw(self, graph, n = 0):
        self._draw(graph, fillcolor = "cyan",
                   style = "filled", shape = "rectangle")

    def kind(self):
        return "nextact"


LOSE = LosingNode("LOSE")
START = WinningNode("START")


class Transition(object):
    def __init__(self, fr, to, label = ""):
        self.fr = fr
        self.to = to
        self.label = label
        self.transforms = []
        self.restrictions = []
        if label:
            for t in parse_rules(label):
                if isinstance(t, (AreTransform, HaveTransform)):
                    self.restrictions.append(t)
                else:
                    self.transforms.append(t)

        self.traversals = 0

        self.is_losing = (isinstance(self.to, LosingNode) or
                          any(isinstance(x, LoseTransform)
                              for x in self.transforms))
        self.is_winning = False

    def __repr__(self):
        arrow = ""
        if self.transforms:
            arrow = "(%s)" % ",".join(repr(x) for x in self.transforms)
        if self.is_winning:
            arrow += "[win]"
        if self.is_losing:
            arrow += "[lose]"
        if self.restrictions:
            arrow += "(%s)" %  ",".join([str(x) for x in self.restrictions])

        arrow = arrow.replace("[", "").replace("]", "")
        return "(%r -%s-> %r)" % (self.fr, arrow, self.to)

    def allowed(self, player):
        for t in self.restrictions:
            if not t.apply(player):
                return False
        return True

    def required(self):
        for t in self.restrictions:
            if not t.optional:
                return True
        return False

    def draw(self, graph, n = 0):
        kw = {"dir": "forward"}
        if isinstance(self.to, LosingNode):
            kw['color'] = "red"
            kw['weight'] = 1
            if self.to is LOSE:
                return
        elif self.to.label.endswith(LOSE.label):
            kw['weight'] = 0
            kw['color'] = "red"
            return
        elif self.is_winning:
            kw['color'] = "#008020"
        elif self.is_losing:
            kw['weight'] = 1
            kw['color'] = "red"

        if n:
            frac = float(self.traversals) / n
            if frac < .01:
                kw['style'] = 'dotted'
                if kw.get('color') == "red" and frac == 0:
                    kw['color'] = 'grey'

            kw['weight'] = int(20 * frac + 1)
            kw['penwidth'] = int(20 * frac + 1)
            if 'color' not in kw and not frac:
                kw['color'] = "grey"
            if frac < 10:
                kw['label'] = "  %3.1f%%  " % (frac * 100)
            else:
                kw['label'] = "  %d%%  " % int(frac * 100)

        if self.restrictions:
            kw['label'] = "".join([kw.get('label', ""),
                                   str(self.restrictions)])
        elif self.transforms:
            kw['label'] = "".join([kw.get('label', ""),
                                   str(self.transforms)])
        if 'label' in kw:
            kw['label'] = kw['label'].replace("[", "").replace("]", "")
            # TODO temp:
            del kw['label']

        to_label = self.to.label
        # make interact transitions nice and fat
        if isinstance(self.fr, NextNode):
            kw['penwidth'] = 20
            kw['weight'] = 1
        edge = pydot.Edge(self.fr.label, to_label, **kw)
        graph.add_edge(edge)

    def apply(self, player):
        self.traversals += 1
        logger.debug(" + Transitioning %r", self)
        for transform in self.transforms:
            transform.apply(player)

    def export(self, act_num, stream):
        if not isinstance(self.to, (NextNode, WinningNode, LosingNode)):
            stream.write("%s, %s, %s, %s\n" %
                (act_num, self.fr.number(), self.to.number(), self.label))


class Act(object):
    def __init__(self, act):
        self.act = act
        self.roman = {'1': "i",
                      '2': "ii",
                      '3': "iii",
                      '4': "iv",
                      '5': "v"}[act]
        # TODO: hack here
        self.nodes = {"LOSE": LOSE}
        self.first_node = None
        self.transitions = []
        self.map = {}
        self.lose_map = {}
        self.tmap = {}
        self.nsimulations = 0

    def make_transitional(self, nname, node_cls):
        # create a new node of this class and COPY
        node = self[nname]
        newnode = node_cls(node.label)
        for k, v in node.__dict__.iteritems():
            setattr(newnode, k, v)

        # replace this node for that node in transitions
        for t in self.tmap.get(node, []):
            if node.label == t.to.label:
                t.to = newnode
            elif node.label == t.fr.label:
                t.fr = newnode

        # replace this node in the lookup
        self.nodes[node.label] = newnode
        return newnode

    def __getitem__(self, k):
        newk = ".".join([self.roman, k])
        if k not in self.nodes and newk not in self.nodes:
            self.nodes[newk] = Node(newk, self.act)
            if k == "1":
                self.first_node = self.nodes[newk]
        return self.nodes.get(newk, self.nodes.get(k))

    def add(self, fr, to, label = None):
        fr = self[fr] if not isinstance(fr, Node) else fr
        to = self[to] if not isinstance(to, Node) else to

        t = Transition(fr, to, label)

        # store all transitions for this node (for updating)
        self.tmap.setdefault(fr, []).append(t)
        self.tmap.setdefault(to, []).append(t)

        # store the existence of this transition (for drawing)
        self.transitions.append(t)

        # lose transitions are "special"
        if isinstance(to, LosingNode) or t.is_losing:
            d = self.lose_map.setdefault(fr, [])
            d.append(t)
        else:
            d = self.map.setdefault(fr, [])
            d.append(t)

    def do_step(self, node, player):
        if node.game is not None:
            # figure out if we can lose everything
            will_lose = any(isinstance(t.to, LosingNode)
                            for t in self.lose_map.get(node, []))
            if node.apply(player, will_lose = will_lose):
                t = choice(self.map[node])
            else:
                # reset the HP to 1 if needs be.
                if player.hp == 0 and not player.use_medipack():
                    player.hp = max(player.hp, 1)
                t = choice(self.lose_map[node])
        else:
            node.apply(player)
            # "enhance" winning paths to improve overall statistics
            for i in range(4):
                allowed = [x for x in (self.map.get(node, []) +
                                       self.lose_map.get(node, []))
                           if x.allowed(player)]
                required = [x for x in allowed if x.required()]
                if required:
                    t = choice(required)
                else:
                    t = choice(allowed)
                # look ahead for gameover and try to avoid
                _node = t.to
                ts = self.map.get(_node, []) + self.lose_map.get(_node, [])
                if len(ts) != 1 or not isinstance(ts[0].to, LosingNode):
                    break
        t.apply(player)

        return t.to

    def add_losing(self, fr):
        self.add(fr, "LOSE")

    def draw(self, path = "/tmp", graph = None):
        save = False
        if graph is None:
            graph = pydot.Dot(graph_type='digraph', fontname="Verdana")
            save = True

        for node in self.nodes.values():
            node.draw(graph, self.nsimulations)

        for transition in self.transitions:
            transition.draw(graph, self.nsimulations)

        if save:
            graph.write_dot(os.path.join(path, "act%s.dot" % self.act))
            graph.write_pdf(os.path.join(path, "act%s.pdf" % self.act))

    def has_losing(self, node):
        if not isinstance(node, Node):
            node = self[node]
        if node in self.map:
            return (any(t.is_losing for t in self.map.get(node, [])) or
                    any(t.is_losing for t in self.lose_map.get(node, [])))
        return True

    def validate(self):
        for node in self.nodes.values():
            if isinstance(node.game, (Battle, Minigame)):
                if not self.has_losing(node):
                    self.add_losing(node)
                for n in self.map.get(node, []):
                    if not n.is_losing:
                        n.is_winning = True

    def randomize(self):
        """
        Randomize the scenes
        """
        nodes = []
        ends = []
        for n in self.nodes.values():
            if not isinstance(n, (NextNode, LosingNode)):
                label, _, _ = n.label.partition(" ")
                label = label.split('.')[-1]
                if n in self.map:
                    if isinstance(self.map[n][0].to, (WinningNode, NextNode)):
                        ends.append(label)
                        continue
                nodes.append(label)
        nodes.append("")
        nodes = list(set(nodes))
        nodes.sort(key = lambda x: (x and int(''.join(y for y in x if y.isdigit())),
                                  ''.join(y for y in x if not y.isdigit())))
        newnums = range(2, len(nodes) + 1)
        shuffle(newnums)
        newnums = [1] + newnums
        res = []
        for x, y in zip(nodes, newnums):
            res.append((self.act, x, y, roll()))
        for i, x in enumerate(ends):
            res.append((self.act, x, len(nodes) + i + 1, roll()))
        res.sort(key = lambda x: (x[1] and int(''.join(y for y in x[1] if y.isdigit())),
                                  ''.join(y for y in x[1] if not y.isdigit())))
        return res

    def simulate(self, player = None, enable_psychic = False):
        if not player:
            player = Player()
        self.nsimulations += 1
        node = self['1']
        breadcrumbs = []
        last_death = None
        psychic_avoids = 10
        while True:
            if not self.map.get(node) and not self.lose_map.get(node):
                return node, player

            new_node = self.do_step(node, player)

            if isinstance(new_node, LosingNode) and not node.game:
                if enable_psychic and len(breadcrumbs) > 1 and psychic_avoids and \
                        "psychic aspect" in player.inventory:
                    psychic_avoids -= 1
                    if psychic_avoids < 7:
                        breadcrumbs.pop()
                        if len(breadcrumbs) > 1 and psychic_avoids < 4:
                            breadcrumbs.pop()
                    node, hp0 = breadcrumbs.pop()
                    player.hp = hp0
                else:
                    return new_node, player
            else:
                breadcrumbs.append((node, player.hp))
                node = new_node

    def export(self, stream):
        # print player
        # print items
        # print specializations
        for s in sorted(Status.seen.values(), key=lambda x: (x.__class__, x.name)):
            s.export(stream)

        # print battles, gains, and resets
        for n in self.nodes.values():
            s = StringIO()
            n.export(s)
            s = s.getvalue()
            if s:
                stream.write("%s, %s\n" % (self.act, s))
        # print special nodes
        seen = set()
        for t in self.transitions:
            if t.to not in seen and \
               isinstance(t.to, (NextNode, WinningNode, LosingNode)) and \
               not t.fr.game:
                seen.add(t.to)
                stream.write("%s, %s, %s\n" %
                    (self.act, t.to.number(), t.to.kind()))

        # print transitions
        for t in self.transitions:
            t.export(self.act, stream)


class ActSimResults(object):
    def __init__(self):
        self.nwin = 0
        self.nlose = 0
        self.nend = 0
        self.nplay = 0

        self.final_hp = []
        self.final_wits = []
        self.final_charisma = []
        self.final_fighting = []
        self.by_class = {k: {} for k in BecomeTransform.classes}
        self.by_weapon = {}

    def track(self, player, final_node):
        # store the player stats
        self.final_hp.append(player.hp)
        self.final_wits.append(player.wits)
        self.final_charisma.append(player.charisma)
        self.final_fighting.append(player.fighting)

        if not isinstance(final_node, (WinningNode, NextNode)):
            if final_node is LOSE:
                self.nlose += 1
                for x in player.specialize:
                    if x in self.by_class:
                        d = self.by_class[x]
                        d['lose'] = d.setdefault('lose', 0) + 1
            else:
                self.nend += 1
                for x in player.specialize:
                    if x in self.by_class:
                        d = self.by_class[x]
                        d['end'] = d.setdefault('end', 0) + 1
        else:
            self.nwin += 1
            for x in player.specialize:
                if x in self.by_class:
                    d = self.by_class[x]
                    d['win'] = d.setdefault('win', 0) + 1

            d = self.by_weapon
            for x in player.inventory:
                d[x] = d.get(x, 0) + 1

    def summarize(self, act_num):
        print "Player Stats: (Act %d) %d wins, %d lose, %d gameover\n  ==> %d total, %5.2f%% loss)" % (
            act_num, self.nwin, self.nlose, self.nend, self.nplay, (100. * self.nlose / max(self.nplay, 1)))
        print "%10s %5.2f +/- %5.2f" % ("hp", mean(self.final_hp),
                                        sigma(self.final_hp))
        print "%10s %5.2f +/- %5.2f" % ("wits", mean(self.final_wits),
                                        sigma(self.final_wits))
        print "%10s %5.2f +/- %5.2f" % ("charisma", mean(self.final_charisma),
                                        sigma(self.final_charisma))
        print "%10s %5.2f +/- %5.2f" % ("fighting", mean(self.final_fighting),
                                        sigma(self.final_fighting))
        print "by class:"
        for x in sorted(self.by_class):
            d = self.by_class[x]
            tot = sum(d.values()) or 1
            print "%10s: %5.2f%% lose %r" % \
               (x, 100 * float(d.get('lose',0)) / tot, d)
#        print sorted(self.by_weapon.iteritems(), key = lambda x: self.by_weapon[x[0]])
        print


class Book(object):
    def __init__(self, acts, player_transforms):
        self.acts = acts
        self.player_transforms = player_transforms

    def new_player(self):
        player = Player()
        for transform in self.player_transforms:
            transform.apply(player)
            player.max_hp = player.hp
        return player

    @classmethod
    def parse_act(cls, actfile, randomizer = "randomize_acts.csv"):

        remap = {}
        if randomizer:
            with open(randomizer) as handle:
                reader = csv.reader(handle)
                for row in reader:
                    act, scene, newscene, _ = row
                    remap[(int(act), scene)] = newscene

        acts = {i: Act(str(i)) for i in range(1, 6)}
        player = []
        with open(actfile) as handle:
            for line in handle:

                bits = [x.strip() for x in line.split(',')]

                # ignore comments
                if line.startswith("#"):
                    continue

                # player info  line
                elif line.startswith("player"):
                    _, bits = line.split(":")
                    player = parse_rules(bits)

                # Item info lines
                elif Item.couldbe(line):
                    Item.parse(line)

                # Specialization lines
                elif Specialization.couldbe(line):
                    Specialization.parse(line)

                # label?
                elif len(bits) == 3:
                    act, node, kind = bits
                    act = int(act[-1])
                    node = remap.get((act, node), node)
                    if Node.is_kind(kind):
                        acts[act][node].set_kind(kind)
                    else:
                        # Game state change (win/lose/next-act)
                        trans = Node.is_transitional(kind)
                        if trans:
                            node = acts[act].make_transitional(node, trans)
                        elif "TK" not in line:
                            logging.error("What? %s", line)

                # transition?
                elif len(bits) >= 4:
                    act, start, end = bits[:3]
                    ops = ",".join(bits[3:])
                    act = int(act[-1])
                    start = remap.get((act, start), start)
                    end = remap.get((act, end), end)
                    acts[act].add(start, end, ops)

        # make sure everything went well
        for act in acts.itervalues():
            act.validate()

        return cls(acts, player)

    def randomize(self):
        import csv
        with open("/tmp/randomize_acts.csv", "w") as handle:
            writer = csv.writer(handle)
            for i in range(len(self.acts)):
                for line in acts[i + 1].randomize():
                    writer.writerow(line)

    def draw_all(self, path = "/tmp", do_cluster = True):
        graph = pydot.Dot(simplify=True, size = "7.5, 10",
                          ratio = "compress",
                          nodesep = "1.5", ranksep = ".25",
                          graph_type='digraph', fontname="Verdana")
        cluster = pydot.Cluster("start") if do_cluster else graph
        for node in (START,):
            node.draw(cluster)
        if do_cluster:
            graph.add_subgraph(cluster)

        for act in reversed(self.acts):
            cluster = pydot.Cluster('Act_%s' % act.act,
                                    label='Act %s' % act.act.upper()) \
                                    if do_cluster else graph
            act.draw(graph = cluster)
            if do_cluster:
                graph.add_subgraph(cluster)

        # draw the start transtion
        Transition(START, self.acts[1].first_node, "").draw(graph)
        # draw the next transitions:
        for i, act in enumerate(self.acts):
            for n in act.nodes.itervalues():
                if isinstance(n, NextNode):
                    Transition(n, self.acts[i + 1].first_node).draw(graph)
        graph.write_pdf(os.path.join(path, "all_acts.pdf"))
        graph.write_dot(os.path.join(path, "all_acts.dot"))

    def monte(self, niter = 100, draw = False, verbose = False, nacts = 5):

        sim_results = {i: ActSimResults() for i in range(1, 6)}

        inventories = {}
        things = {}
        specializes = {}

        for i in xrange(niter):
            player = self.new_player()
            cur = 1
            while True:
                cur_act = self.acts[cur]

                node, player = cur_act.simulate(player)
                sim_result[cur].track(player, node)

                if not isinstance(node, (WinningNode, NextNode)):
                    logger.debug("#%d: LOST Act %s", i + 1, cur_act.act)
                    break
                else:
                    if isinstance(node, WinningNode):
                        logger.debug("SWEET VICTORY IS MINE!")
                        break
                    # CHANGING ACTS!
                    if isinstance(node, NextNode) and cur < nacts:
                        cur += 1
                        player.hp = player.max_hp
                        if "engineer" in player.specialize:
                            player.wits += 2
                        elif "fighter" in player.specialize:
                            player.hp += 2
                            player.max_hp += 2
                            player.fighting += 1
                        elif "medic" in player.specialize:
                            player.charisma += 1
                            player.wits += 1
                        else:
                            raise "oh shit"
                        assert player.hp
                    else:
                        #print node, cur
                        logger.debug("#%d: Completed Act %s", i + 1, cur_act.act)
                        break

            i = tuple(sorted(player.inventory.keys()))
            s = tuple(sorted(player.specialize.keys()))
            for x in i:
                things[x] = things.get(x, 0) + 1
            inventories[i] = inventories.get(i, 0) + 1
            specializes[s] = specializes.get(s, 0) + 1

        if verbose:
            print "\nInventories:"
            for i in sorted(inventories, key = lambda x: inventories[x],
                            reverse = True):
                print "%7.2f%% %s" % (inventories[i] * 100. / niter, i)

            print "\nThings:"
            for i in sorted(things, key = lambda x: things[x],
                            reverse = True):
                print "%7.2f%% %s" % (things[i] * 100. / niter, i)

            print "\nSpecializations:"
            for s in sorted(specializes, key = lambda x: specializes[x],
                            reverse = True):
                print "%7.2f%% %s" % (specializes[s] * 100. / niter, s)
            print

        if draw:
            self.draw()

        for i, sim_result in sim_results.iteritems():
            sim_result.summarize(i)


def do(n = 10, draw = False, verbose = False, nacts = 3):
    book = Book.parse_act("all_acts_zero_start.txt")
    book.monte(n, verbose = verbose, nacts = nacts)
    if verbose:
        Battle.summarize()
    if draw:
        for i, act in acts.iteritems():
            if i <= nacts:
                act.draw()

if __name__ == "__main__":
    do()
