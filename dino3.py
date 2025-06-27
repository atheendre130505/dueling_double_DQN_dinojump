import pygame
import random
import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_Y = 350
FPS = 60
GRAVITY = 1
JUMP_VELOCITY = -15
GAME_SPEED_START = 6
SPEED_INCREMENT = 0.0
MIN_OBSTACLE_GAP = 30
MAX_OBSTACLE_GAP = 100
BIRD_SPEED_THRESHOLD = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Dinosaur(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.width, self.height = 20, 40
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect(x=50, bottom=GROUND_Y)
        self.vel_y = 0
        self.is_jumping = False
        self.is_ducking = False

    def update(self):
        if self.is_jumping:
            self.vel_y += GRAVITY
            self.rect.y += self.vel_y
            if self.rect.bottom >= GROUND_Y:
                self.rect.bottom = GROUND_Y
                self.is_jumping = False
                self.vel_y = 0

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self.is_jumping = True
            self.vel_y = JUMP_VELOCITY

    def duck(self):
        if not self.is_jumping and not self.is_ducking:
            self.is_ducking = True
            size = self.width
            self.image = pygame.Surface((size, size))
            self.image.fill(GREEN)
            x = self.rect.x
            self.rect = self.image.get_rect(x=x, bottom=GROUND_Y)

    def stand_up(self):
        if self.is_ducking:
            self.is_ducking = False
            self.image = pygame.Surface((self.width, self.height))
            self.image.fill(GREEN)
            x = self.rect.x
            self.rect = self.image.get_rect(x=x, bottom=GROUND_Y)

    def reset(self):
        self.is_jumping = False
        self.is_ducking = False
        self.vel_y = 0
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect(x=50, bottom=GROUND_Y)

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, speed, obstacle_type='cactus'):
        super().__init__()
        self.type = obstacle_type
        if obstacle_type == 'cactus':
            w, h = 20, random.randint(20, 50)
            self.image = pygame.Surface((w, h))
            self.rect = self.image.get_rect(x=SCREEN_WIDTH, bottom=GROUND_Y)
        else:
            w, h = 30, 20
            y = random.choice([50, 80, 110])
            self.image = pygame.Surface((w, h))
            self.rect = self.image.get_rect(x=SCREEN_WIDTH, y=GROUND_Y - y)
        self.image.fill(RED)
        self.speed = speed

    def update(self):
        self.rect.x -= self.speed
        if self.rect.right < 0:
            self.kill()

class DinoGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pygame Dino Run")
        self.clock = pygame.time.Clock()
        self.dino = Dinosaur()
        self.obstacles = pygame.sprite.Group()
        self.sprites = pygame.sprite.Group(self.dino)
        self.score = 0
        self.high_score = 0
        self.speed = GAME_SPEED_START
        self.game_over = False
        self.timer = 0
        self.next_gap = random.randint(MIN_OBSTACLE_GAP, MAX_OBSTACLE_GAP)
        self.font = pygame.font.Font(None, 36)

    def spawn(self):
        if self.speed >= BIRD_SPEED_THRESHOLD%2:
            choice = random.choices(['cactus', 'bird'], weights=(6,4))[0]
        else:
            choice = 'cactus'
        obs = Obstacle(self.speed, choice)
        self.obstacles.add(obs)
        self.sprites.add(obs)

    def reset(self):
        self.dino.reset()
        self.obstacles.empty()
        self.sprites = pygame.sprite.Group(self.dino)
        self.score = 0
        self.speed = GAME_SPEED_START
        self.game_over = False
        self.timer = 0
        self.next_gap = random.randint(MIN_OBSTACLE_GAP, MAX_OBSTACLE_GAP)


    def update(self):
        if self.game_over:
            return
        self.score += 1
        if self.score % 100 == 0:
            self.speed += SPEED_INCREMENT
        self.timer += 1
        if self.timer >= self.next_gap:
            self.spawn()
            self.timer = 0
            self.next_gap = random.randint(MIN_OBSTACLE_GAP, MAX_OBSTACLE_GAP)
        self.sprites.update()
        if pygame.sprite.spritecollideany(self.dino, self.obstacles):
            self.game_over = True
            self.high_score = max(self.high_score, self.score)

    def draw(self):
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        self.sprites.draw(self.screen)
        s_text = self.font.render(f"Score: {self.score}", True, BLACK)
        h_text = self.font.render(f"High: {self.high_score}", True, BLACK)
        self.screen.blit(s_text, (SCREEN_WIDTH - 150, 10))
        self.screen.blit(h_text, (SCREEN_WIDTH - 150, 40))
        if self.game_over:
            over = self.font.render("GAME OVER! SPACE/UP to restart", True, BLACK)
            r = over.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(over, r)
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_SPACE, pygame.K_UP):
                        if self.game_over:
                            self.reset()
                        else:
                            self.dino.jump()
                    elif e.key == pygame.K_DOWN:
                        self.dino.duck()
                elif e.type == pygame.KEYUP:
                    if e.key == pygame.K_DOWN:
                        self.dino.stand_up()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


class DinoEnv(DinoGame):
    def __init__(self):
        super().__init__()
        self.action_space = [0,1,2] # 0: jump, 1: duck, 2: do nothing
        self.observation_space = (SCREEN_WIDTH, SCREEN_HEIGHT, 3)
        self.passed_obstacles = set()
        
    def step(self, action):
        # apply action…
        if action == 0:
            self.dino.jump()
        elif action == 1:   
            self.dino.duck()
        elif action == 2:
            self.dino.stand_up()

        self.update()
        obstacles_passed = 0
        reward = 0
        done   = False

        if self.game_over:
            done   = True
            reward = -100

        else:
            # New: reward +10 each time an obstacle passes left of Dino
            for obs in list(self.obstacles):
                if obs.rect.x < self.dino.rect.x and obs not in self.passed_obstacles:
                    self.passed_obstacles.add(obs)
                    reward += 10
                    obstacles_passed+=1
            reward+=1
        return self.get_state(), reward, done
            
    
    
    def reset(self):
        super().reset()
        self.passed_obstacles.clear()   # ← New
        return self.get_state()

    
    def get_state(self):
        next_obs = None
        min_dist = float('inf')

        for obs in self.obstacles:
            dist = obs.rect.x - self.dino.rect.x
            if dist > 0 and dist < min_dist:
                min_dist = dist
                next_obs = obs

        if next_obs:
            obs_x = next_obs.rect.x
            obs_h = next_obs.rect.height
            obs_w = next_obs.rect.width
            is_bird = 1.0 if next_obs.type == 'bird' else 0.0
            obs_speed = next_obs.speed / 20  # Assuming speed caps around 20
            vert_dist = (self.dino.rect.y - next_obs.rect.y) / SCREEN_HEIGHT if is_bird else 0.0
        else:
            obs_x = SCREEN_WIDTH
            obs_h = 0
            obs_w = 0
            is_bird = 0.0
            obs_speed = 0.0
            vert_dist = 0.0

        state = np.array([
            self.dino.rect.y / SCREEN_HEIGHT,                        # Dino Y-position (0 = ground)
            float(self.dino.is_jumping),                             # Jumping flag
            (obs_x - self.dino.rect.x) / SCREEN_WIDTH,              # Distance to obstacle (normalized)
            obs_h / 100,                                             # Obstacle height (max ~100)
            self.speed / 20,                                         # Speed (max expected 20)
            is_bird,                                                 # 1 if bird, 0 if cactus
            obs_speed,                                               # Obstacle speed normalized
            vert_dist,                                               # Vertical gap (useful only for birds)
            obs_w / 30                                               # Obstacle width (max ~30)
        ], dtype=np.float32)

        return state

 

    
if __name__ == "__main__":
    DinoGame().run()
