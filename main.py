from data_compiler import DataCompiler
from play_analyzer import PlayAnalyzer
from annotated_play import AnnotatedPlay
from annotated_field_utils import AnnotatedField
import random

dc = DataCompiler()

game = random.choice(dc.game_ids)
play_ids = dc.get_play_ids_from_game_id(game)
pid = random.choice(play_ids)
info = dc.get_play_data_by_id(game, pid)

# get a random play with Cover 1, 2, 3 or quarters coverage that was a passing play
print("Selecting play...")
while info['pff_passCoverage'].iloc[0] not in ["Cover-1", "Cover-2", "Cover-3", "Quarters"] or (info['passResult'].iloc[0] != "C" and info['passResult'].iloc[0] != "I"):
    game = random.choice(dc.game_ids)
    play_ids = dc.get_play_ids_from_game_id(game)
    pid = random.choice(play_ids)
    info = dc.get_play_data_by_id(game, pid)
print(f"Play selected. gid {game} pid {pid}")

play = dc.get_play_tracking_by_id(game, pid)
play_info = dc.get_play_data_by_id(game, pid)
analyzer = PlayAnalyzer()
annotated_field = AnnotatedField()
an_play = AnnotatedPlay(dc, analyzer, game, pid)
annotated_field.plot_play_animation(an_play)  # display a field window with predictions