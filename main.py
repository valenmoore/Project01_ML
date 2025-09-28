from data_compiler import DataCompiler
from play_analyzer import PlayAnalyzer
from annotated_play import AnnotatedPlay
from annotated_field_utils import AnnotatedField
import random

dc = DataCompiler()

select_random = True

print("Selecting play...")
if select_random:
    game = random.choice(dc.game_ids)
    play_ids = dc.get_play_ids_from_game_id(game)
    pid = random.choice(play_ids)
    info = dc.get_play_data_by_id(game, pid)

    # get a random play with Cover 1, 2, 3 or quarters coverage that was a passing play
    while info['pff_passCoverage'].iloc[0] not in ["Cover-1", "Cover-2", "Cover-3", "Quarters"] or (info['passResult'].iloc[0] != "C" and info['passResult'].iloc[0] != "I"):
        game = random.choice(dc.game_ids)
        play_ids = dc.get_play_ids_from_game_id(game)
        pid = random.choice(play_ids)
        info = dc.get_play_data_by_id(game, pid)
else:
    # good_plays = [(2022100911, 1646), (2022101000, 1451), (2022100910, 3396), (2022100909, 3244)]
    good_plays = [(2022100911, 1646), (2022100600, 1704), (2022100913, 172), (2022101000, 287), (2022100913, 559), (2022100910, 3396), (2022100902, 2110)]
    index = 0
    game = good_plays[index][0]
    pid = good_plays[index][1]
print(f"Play selected. gid {game} pid {pid}")

play = dc.get_play_tracking_by_id(game, pid)
play_info = dc.get_play_data_by_id(game, pid)
analyzer = PlayAnalyzer()
annotated_field = AnnotatedField()
an_play = AnnotatedPlay(dc, analyzer, game, pid)
annotated_field.plot_play_animation(an_play)  # display a field window with predictions