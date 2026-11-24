"""The 54 in-scope effects (both decks) written as plain data using the block vocabulary."""
import json

# ---- tiny helpers so the data stays readable
def F(**kw): return {"block": "filter", **kw}
def Fx(n): return {"block": "fixed", "n": n}
def Per(each, counter): return {"block": "per_count", "each": each, "counter": counter}
def Sm(*parts): return {"block": "sum", "parts": list(parts)}
def C(name, arg=None, negate=False): return {"block": "cond", "name": name, "arg": arg, "negate": negate}
def IT(cond, then, otherwise=None): return {"block": "if_then", "condition": cond, "then": then, **({"otherwise": otherwise} if otherwise else {})}
def T(which="this_pokemon", side="you", filter=None): return {"block": "target", "which": which, "side": side, **({"filter": filter} if filter else {})}
def Mv(source, destination, **kw): return {"block": "move_cards", "source": source, "destination": destination, **kw}
def Dr(count, side="you"): return {"block": "draw", "count": count if isinstance(count, dict) else Fx(count), "side": side}
def Sw(side="you", target="your_active"): return {"block": "switch_active", "side": side, "target": target}
def Usage(scope="none", key=None, first=False): return {"block": "usage", "scope": scope, "key": key, "first_turn_only": first}
def E(move_id, timing, **kw): return {"block": "effect", "move_id": move_id, "timing": timing, **kw}

POKEMON = ("Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon")
BASIC_F_ENERGY = F(card_kind=("Basic Energy",), type="{F}")
ONCE = Usage("per_pokemon")

EFFECTS = [
 # ---------------- Energy
 E("13_0","on_attach_from_hand",steps=[Dr(4)],notes="Enriching Energy; also provides {C} (see energy_provides)"),
 E("11_0","passive",steps=[{"block":"prevent","what":"attack_effects","to":"attached_pokemon","source":"opponent_attacks"}],notes="Mist Energy; also provides {C}"),
 # ---------------- Pokémon: attacks
 E("848_0","attack",steps=[Sw("you","this_pokemon")],notes="Buneary, Run Around"),
 E("848_1","attack",notes="Buneary, Kick: plain damage"),
 E("209_1","attack",steps=[Mv("attached_to_this","hand",filter=F(family="energy"),count=1)],notes="Chien-Pao, Icicle Loop"),
 E("66_1","attack",notes="Dudunsparce, Land Crush: plain damage"),
 E("65_0","attack",notes="Dunsparce, Gnaw: plain damage"),
 E("305_0","attack",steps=[Sw("you","this_pokemon")],notes="Dunsparce, Trading Places"),
 E("65_1","attack",steps=[{"block":"coin_flip","heads":[{"block":"prevent","what":"damage_and_effects","to":"this_pokemon","source":"opponent_attacks","duration":"opponent_next_turn"}]}],notes="Dunsparce, Dig"),
 E("305_1","attack",notes="Dunsparce, Ram: plain damage"),
 E("174_1","attack",works_only_if=[C("stadium_in_play")],notes="Fan Rotom, Assault Landing"),
 E("674_1","attack",steps=[{"block":"damage_self","amount":70}],notes="Hariyama, Wild Press"),
 E("675_1","attack",notes="Lunatone, Power Gem: plain damage"),
 E("673_0","attack",notes="Makuhita, Corkscrew Punch: plain damage"),
 E("673_1","attack",notes="Makuhita, Confront: plain damage"),
 E("861_0","attack",damage=Per(50,"opp_hand_size"),notes="Mega Froslass ex, Resentful Refrain"),
 E("861_1","attack",steps=[{"block":"apply_status","status":"Asleep","target":T("opponent_active","opponent")}],notes="Mega Froslass ex, Absolute Snow"),
 E("849_0","attack",damage=Sm(Fx(60),IT(C("moved_to_active_this_turn"),Fx(170))),notes="Mega Lopunny ex, Gale Thrust"),
 E("849_1","attack",ignore_defender_effects=True,notes="Mega Lopunny ex, Spiky Hopper"),
 E("678_0","attack",steps=[Mv("discard","attached_to_bench",filter=BASIC_F_ENERGY,count=3,up_to=True,distribution="any")],notes="Mega Lucario ex, Aura Jab"),
 E("678_1","attack",steps=[{"block":"lockout","attack":"this_attack","duration":"your_next_turn"}],notes="Mega Lucario ex, Mega Brave"),
 E("1071_1","attack",steps=[Mv("this_pokemon_and_attached","hand",count="all")],notes="Meowth ex, Tuck Tail"),
 E("858_1","attack",notes="Psyduck, Ram: plain damage"),
 E("677_0","attack",steps=[{"block":"lockout","attack":"this_attack","duration":"your_next_turn"}],notes="Riolu, Accelerating Stab"),
 E("860_0","attack",notes="Snorunt, Chilly: plain damage"),
 E("676_0","attack",works_only_if=[C("has_on_bench","Lunatone")],ignore_weakness_resistance=True,notes="Solrock, Cosmic Beam"),
 # ---------------- Pokémon: abilities
 E("209_0","on_play_bench",optional=True,steps=[Mv("stadium_in_play","discard")],notes="Chien-Pao, Snow Sink"),
 E("66_0","activated",usage=ONCE,optional=True,steps=[Dr(3),{"block":"if","condition":C("drew_any"),"then":[Mv("this_pokemon_and_attached","deck_shuffle",count="all")]}],notes="Dudunsparce, Run Away Draw"),
 E("174_0","activated",usage=Usage("per_name","Fan Call",True),optional=True,steps=[Mv("deck","hand",filter=F(card_kind=POKEMON,type="{C}",hp_max=100),count=3,up_to=True,reveal=True,shuffle_after=True)],notes="Fan Rotom, Fan Call"),
 E("674_0","on_evolve",usage=ONCE,optional=True,steps=[Sw("opponent")],notes="Hariyama, Heave-Ho Catcher"),
 E("675_0","activated",usage=Usage("per_name","Lunar Cycle"),optional=True,requires=[C("has_in_play","Solrock")],cost=[Mv("hand","discard",filter=BASIC_F_ENERGY,count=1)],steps=[Dr(3)],notes="Lunatone, Lunar Cycle"),
 E("1071_0","on_play_bench",usage=Usage("per_name_contains","Last-Ditch"),optional=True,steps=[Mv("deck","hand",filter=F(card_kind=("Supporter",)),count=1,reveal=True,shuffle_after=True)],notes="Meowth ex, Last-Ditch Catch; once during your turn"),
 E("858_0","passive",status="no_effect_in_scope",notes="Psyduck, Damp: removes Abilities that need the Pokémon to Knock Itself Out; no card in either deck has one"),
 # ---------------- Trainers
 E("1174_0","passive",steps=[{"block":"modify_stat","stat":"retreat_cost","amount":-2,"side":"attached"}],notes="Air Balloon (Tool)"),
 E("1264_0","passive",steps=[{"block":"prevent","what":"damage_counters","to":"all_benched","source":"opponent_attacks_and_abilities"}],notes="Battle Cage (Stadium)"),
 E("1182_0","trainer_play",steps=[Sw("opponent")],notes="Boss's Orders"),
 E("1086_0","trainer_play",steps=[Mv("deck","bench",filter=F(card_kind=("Basic Pokémon",),hp_max=70),count=2,up_to=True,shuffle_after=True)],notes="Buddy-Buddy Poffin; puts as many as fit on the Bench (up to 2), because the Bench max of 5 always applies. Project decision, confirm with a player"),
 E("1142_0","trainer_play",steps=[Mv("deck","hand",filter=F(any_of=[F(card_kind=("Basic Energy",),type="{F}"),F(card_kind=("Basic Pokémon",),type="{F}")]),count=1,reveal=True,shuffle_after=True)],notes="Fighting Gong"),
 E("1252_0","passive",steps=[{"block":"modify_stat","stat":"hp","amount":-30,"applies_to":F(card_kind=("Stage 2 Pokémon",)),"side":"both"}],notes="Gravity Mountain (Stadium)"),
 E("1225_0","trainer_play",steps=[Mv("deck","hand",filter=F(card_kind=("Stage 1 Pokémon","Stage 2 Pokémon")),count=1,reveal=True),Mv("deck","hand",filter=F(family="energy"),count=1,reveal=True,shuffle_after=True)],notes="Hilda"),
 E("1213_0","trainer_play",steps=[Mv("hand","deck_shuffle",count="all",side="both"),Dr(4,"both")],notes="Judge"),
 E("1227_0","trainer_play",steps=[Mv("hand","deck_shuffle",count="all"),Dr(IT(C("prizes_remaining_equals",6),Fx(8),Fx(6)))],notes="Lillie's Determination"),
 E("1237_0","trainer_play",status="complex",steps=[Mv("hand","deck_bottom",count="all",side="both"),{"block":"if","condition":C("either_player_put_cards_on_bottom"),"then":[{"block":"for_each_player","steps":[{"block":"coin_flip","side":"that_player","heads":[Dr(6,"that_player")],"tails":[Dr(3,"that_player")]}]}]}],notes="Lucian: each player flips a coin; needs per-player coin flips"),
 E("1097_0","trainer_play",steps=[Mv("discard","hand",filter=F(any_of=[F(family="pokemon"),F(card_kind=("Basic Energy",))]),count=1)],notes="Night Stretcher"),
 E("1152_0","trainer_play",steps=[Mv("deck","hand",filter=F(family="pokemon",rule_box=False),count=1,reveal=True,shuffle_after=True)],notes="Poké Pad"),
 E("1122_0","trainer_play",steps=[Mv("top_of_deck","hand",look_n=7,filter=F(card_kind=("Supporter",)),count=1,optional=True,reveal=True,rest_to="deck_shuffle")],notes="Pokégear 3.0"),
 E("1141_0","trainer_play",steps=[{"block":"modify_stat","stat":"attack_damage","amount":30,"applies_to":F(family="pokemon",type="{F}"),"side":"you","duration":"this_turn"}],notes="Premium Power Pro"),
 E("1413_0","trainer_play",requires=[C("opponent_prizes_at_most",3)],steps=[Mv("hand","deck_bottom",count="all",side="opponent"),{"block":"if","condition":C("they_put_cards_on_bottom"),"then":[Dr(3,"opponent")]}],notes="Special Red Card"),
 E("1123_0","trainer_play",steps=[Sw("you","your_active")],notes="Switch"),
 E("1219_0","trainer_play",steps=[Mv("deck","hand",filter=F(family="trainer"),count=1,reveal=True,shuffle_after=True)],notes="Team Rocket's Petrel"),
 E("1137_0","trainer_play",steps=[Mv("attached_tool_any","discard",filter=F(card_kind=("Pokémon Tool",)),count=2,up_to=True,side="both")],notes="Tool Scrapper"),
 E("1121_0","trainer_play",cost=[Mv("hand","discard",count=2,others_only=True)],steps=[Mv("deck","hand",filter=F(family="pokemon"),count=1,reveal=True,shuffle_after=True)],notes="Ultra Ball"),
 E("1080_0","trainer_play",requires=[C("own_pokemon_knocked_out_last_opponent_turn")],steps=[Mv("hand","deck_shuffle",count="all",side="both"),Dr(5,"you"),Dr(2,"opponent")],notes="Unfair Stamp"),
 E("1229_0","trainer_play",steps=[{"block":"heal","target":T("chosen","you",F(is_mega=True)),"amount":"all"},{"block":"if","condition":C("healed_any"),"then":[Mv("attached_to_target","hand",filter=F(family="energy"),count="all")]}],notes="Wally's Compassion"),
]
if __name__ == "__main__":
    print(len(EFFECTS), "effects")