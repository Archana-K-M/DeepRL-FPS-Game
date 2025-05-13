from vizdoom import DoomGame, Mode, ScreenResolution
import time

game = DoomGame()
game.load_config("scenarios/deathmatch.cfg")
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.init()

for i in range(5):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        action = [0, 0, 1]  # Shoot
        reward = game.make_action(action)
        print(f"Reward: {reward}")
        time.sleep(0.02)

game.close()