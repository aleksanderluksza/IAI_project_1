"""
In order to run this file:

Install pygame or pygame-ce using pip 

There have been some issues relating to the python version in previous tutorials. 
This code has been validated on windows 11 using python 3.13.15 and regular pygame.

Save the supplied game rules as halma.py in the same folder.
Run:     python halma_pygame_4players.py


"""
from typing import Dict, List, Optional, Tuple, Callable
import pygame
import treelib as tr
import traceback
from copy import deepcopy

from AI_Player_Team27 import (
    # bot_data,
    AI_Player_Team27
)
from halma import (
    check_legal_move,
    check_win_condition,
    move,
    parse_position,
    random_bot,
    win_cells_all,
    initial_pos,
    illegal_bot,
)

MAXIMUM_MOVE_LIMIT: int = 100  # Total turns across all four players, including skipped turns.
VISUALIZE_SEARCH_TREE: bool = True # This is here to match the function signature, the random bot does not visualize anything 
BOT_DELAY_MS: int = 450

INITIAL_BOARD: List[List[int]] = initial_pos

CELL_SIZE = 90
BOARD_X = 55
BOARD_Y = 135
WINDOW_SIZE = (620, 790)
COLORS = {
    1: (200, 45, 55), 2: (40, 100, 210),
    3: (240, 193, 35), 4: (35, 140, 75),
}
TEXT_COLORS = {**COLORS, 3: (145, 105, 0)}  # Darker yellow for readable text.
ZONE_COLORS = {
    1: (250, 215, 218), 2: (210, 225, 250),
    3: (255, 241, 185), 4: (209, 239, 218),
}
GAME_END_COLOR = (117, 65, 165)
NEUTRAL_COLOR = (30, 39, 55)

BotFunction = Callable[
    [List[List[int]], int, bool],
    Tuple[str, str]
]

# def AI_Player_Team27(board: List[List[int]], player: int, visualize: bool = False, depth:int = 2) -> Tuple[str, str]:
#     if len(board) != 5 or len(board[0]) != 5:
#         raise ValueError("Board must be 5 by 5")
#     if player not in [1, 2, 3, 4]:
#         raise ValueError(f"Player {player} is not a valid player")
#     if depth < 1:
#         raise ValueError("Depth must be at least 1")
#     board_map = {}
#     move_tree = tr.Tree()
#     generate_tree(board, player, board_map, depth*4+1, move_tree)
    
#     # after tree is generated and printed, evauluate all paths.
#     leaves = move_tree.leaves()
#     for leaf in leaves:
#         board = leaf.data.board
#         happiness = leaf.data.happiness
#         for player in range(1,5):
#             points_a = 0
#             points_b = 0
#             for row in range(5):
#                 for column in range(5):
#                     if board[row][column] != player: continue
#                     if (row,column) in win_cells_all[player]: points_a += 250 # 25 points for being in the end zone
#                     else:
#                         for x in range(3):
#                             if row == get_player_start_cells(player)[x][0] and column == get_player_start_cells(player)[x][1]: 
#                                 points_b -= 500
#                                 #print(f"Player {player} has a piece in the starting zone at {row},{column}, subtracting 100 points")
#                             else:
#                                 points_b += 100 - 5*(abs(row - win_cells_all[player][x][0]) + abs(column - win_cells_all[player][x][1]))
                            
#                     #print(f"Player {player} has {points_a} points for being in the end zone and {points_b} points for being closer to the end zone than the start zone")
#                     happiness[player-1] = points_a + points_b
#                     temp = leaf.data
#                     temp.happiness = happiness
#                     leaf.data = temp
#     # all leaves have been evaluated for each player, now we need to propagate it up the tree 
#     ancestors = get_ancestors(leaves, move_tree)
#     while ancestors != []:
#         for ancestor in ancestors:
#             if ancestor.identifier == "root": continue
#             children = move_tree.children(ancestor.identifier)
#             if not children:
#                 continue
#             mover = int(children[0].identifier.split(":")[0])-1
#             if mover not in [0,1,2,3]:
#                 raise ValueError(f"Invalid mover {mover} for ancestor {ancestor.identifier}")
#             best_child = max(children,key=lambda child: child.data.happiness[mover])
#             ancestor.data.happiness = list(best_child.data.happiness)
#         ancestors = get_ancestors(ancestors, move_tree) # move up the tree after finishing calculations of current level
#     #pick the best path
#     children = move_tree.children("root")
#     #move_tree.show(line_type="ascii-emv",data_property="happiness")
#     #move_tree.show(line_type="ascii-emv")
#     best_child = max(children, key=lambda child: child.data.happiness[player-1])._identifier
#     move = best_child[2:].replace(">","").replace("(","").replace(")","").replace(" ","").replace(",","").split("-")
#     co: List[int] = [int(move[0][0]),int(move[0][1]),int(move[1][0]),int(move[1][1])]
#     #print(co)
#     old_ref: str = chr(ord("A") + co[1]) + str(co[0] + 1)
#     new_ref: str = chr(ord("A") + co[3]) + str(co[2] + 1)
#     #print(f"Player {player} selected move: {old_ref} -> {new_ref}")
    
#     return old_ref, new_ref #assumes that the identifier of the best child is in the correct format

    ## TODO: propagate happiness values up the tree to the root.
    

    # 1. AI should select moves that maximize its changes of winning.
    # 2. AI should account for other players happiness (no player will choose a move that gives them no chances of winning)
    # 3. If winning is impossible, AI should prioritize moves that maximize its chances of tying.
    # 4. If multiple paths to victory are possible, prioritize the path that is the shortest and gives it the biggest chance of winning
    #       account for other players happiness.
    # 5. If None apply, AI should be as annoying to other players as possible.

    # TODO per project instructions:
    # 1. Implement pruning of repeated positions in the tree. (maybe done, check line 111)
    # 2. Implement branch ordering using heuristics.
    # 3. Implement Alpha-Beta pruning to avoid searching hopeless branches.
    # 4. When the binary flag is set the code should visualize the search tree using treelib and output the result to Team<X>_Tree.png


# # recursivly generate a tree of possible moves for current and next players.
# def generate_tree(board: List[List[int]], player: int, board_map, depth: int,move_tree:tr.Tree,parent = "root", org_depth:int = 0) -> None:
#     if org_depth == 0:
#         org_depth = depth
#     if "root" not in move_tree: #first iteration (first call)
#         temp = bot_data()
#         temp.board = board
#         temp.player = player
#         temp.happiness = [0,0,0,0]
#         temp.board_string = board_to_string(board)
#         move_tree.create_node(
#             "root", "root", data=temp
#         )
#         board_map["root"] = board
#     if depth == 0: #required depth reached, going up
#         return None

#     possible_moves = generate_possible_moves(board, player)
#     for old_pos,new_pos in possible_moves: 
#         board_copy = deepcopy(board)
#         move(board_copy, old_pos, new_pos, player)
#         move_id = f"{player}:{old_pos}->{new_pos}"
#         if (move_id in move_tree) or (board_copy in board_map.values()):
#             if move_tree.get_node(move_id) is None:
#                 continue
#             if move_tree.depth(move_id) > org_depth-depth:
#                 #print(f"Move {move_id} already exists in tree and has been moved from depth {move_tree.depth(move_id)} to {depth}")
#                 #print(board_to_string(board_copy))
#                 move_tree.move_node(move_id, parent)
#                 pass
#             continue

#         temp = bot_data()
#         temp.board = board_copy
#         temp.player = player
#         temp.happiness = [0,0,0,0]
#         temp.board_string = board_to_string(board_copy)
#         move_tree.create_node(move_id, move_id, parent,temp)
#         board_map[move_id] = board_copy
#         generate_tree(board_copy, player%4+1, board_map, depth-1, move_tree, move_id, org_depth)

# def get_player_start_cells(player: int) -> List[Tuple[int,int]]:
#     if player not in [1, 2, 3, 4]:
#         raise ValueError(f"Player {player} is not a valid player")
#     player = player + 2
#     if player > 4: player = player - 4
#     return win_cells_all[player]

# # generates all possible moves for given player and board.
# def generate_possible_moves(board: List[List[int]],player) -> List[Tuple[int,int]]:
#     possible_moves = []
#     for row in range(5):
#         for column in range(5):
#             if board[row][column] != player : continue
#             old_pos = (row,column)
#             if old_pos in win_cells_all[player]: continue # don't move pieces that are already in the end zone
#             for newRow in range(-4,4):
#                 for newColumn in range(-4,4):
#                     new_pos = (row+newRow,column+newColumn)
#                     if new_pos[0] < 0 or new_pos[0] > 4 or new_pos[1] < 0 or new_pos[1] > 4: continue
#                     if not check_legal_move(board,old_pos,new_pos): continue
#                     old_distance = min(
#                         abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])
#                         for target in win_cells_all[player]
#                     )
#                     new_distance = min(
#                         abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
#                         for target in win_cells_all[player]
#                     )
#                     if new_distance-1 > old_distance: continue
#                     possible_moves.append((old_pos,new_pos))
    
#     return possible_moves

# # serializes the board into an easy printable string.
# def board_to_string(board: List[List[int]]) -> str:
#     returnable:str = ""
#     for row in board:
#         for cell in row:
#             returnable += str(cell).replace("0",".")
#         returnable += "\n"
#     return returnable

# def get_ancestors(node_list: List[tr.Node], tree: tr.Tree) -> List[tr.Node]:
#     ancestors = []
#     try:
#         for node in node_list:
#             id = tree.ancestor(node.identifier)
#             if id is None: continue
#             node = tree.get_node(id)
#             if node is None: continue
#             ancestors.append(node)
#         return list(set(ancestors))
#     except:
#         return []

'''
Import your team's function above, then replace random_bot for that player.
Example: from AI_Player_Team1 import AI_Player_Team1
         1: AI_Player_Team1,

Alternatively you can set a desired player's bot function to None and play them yourself
'''
BOT_FUNCTIONS: Dict[int, Optional[BotFunction]] = {
    1: AI_Player_Team27,
    2: AI_Player_Team27,
    3: AI_Player_Team27,
    4: AI_Player_Team27,
}


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The code bellow contains only parts related to the GUI. You do not need to interact with anything down there, your only point of interaction with this file is setting your bot function

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

class HalmaGame:
    def __init__(self, bot_functions: Optional[Dict[int, Optional[BotFunction]]] = None) -> None:
        self.bot_functions = dict(BOT_FUNCTIONS if bot_functions is None else bot_functions)
        if set(self.bot_functions) != {1, 2, 3, 4}:
            raise ValueError("Configure exactly players 1, 2, 3 and 4")
        if any(bot is not None and not callable(bot) for bot in self.bot_functions.values()):
            raise TypeError("Each player must have a bot function or None for a human")
        self.reset()

    def reset(self) -> None:
        self.board = [row[:] for row in INITIAL_BOARD]
        self.current_player = 1
        self.move_count = 0
        self.selected: Optional[Tuple[int, int]] = None
        self.message = "Bots play automatically; use None for a human player."
        self.message_color = NEUTRAL_COLOR
        self.result = check_win_condition(
            self.board, self.move_count, MAXIMUM_MOVE_LIMIT, True
        )
        self.bot_due = pygame.time.get_ticks() + BOT_DELAY_MS

    def destinations(self, old_position: Tuple[int, int]) -> List[Tuple[int, int]]:
        return [
            (row, column)
            for row in range(5)
            for column in range(5)
            if check_legal_move(self.board, old_position, (row, column))
        ]

    def finish_turn(self, message: str, player: int) -> None:
        self.move_count += 1
        self.selected = None
        self.message = message
        self.message_color = TEXT_COLORS[player]
        self.result = check_win_condition(
            self.board, self.move_count, MAXIMUM_MOVE_LIMIT, True
        )
        if self.result.status == "ongoing":
            self.current_player = self.current_player % 4 + 1
        self.bot_due = pygame.time.get_ticks() + BOT_DELAY_MS

    def skip_turn(self, reason: str) -> None:
        player = self.current_player
        print(f"Player {player}: {reason}; turn skipped.")
        self.finish_turn(f"Player {player}: {reason}. Turn skipped.", player)

    def attempt_move(
        self, old_position: Tuple[int, int], new_position: Tuple[int, int]
    ) -> bool:
        if self.result.status != "ongoing":
            return False
        if not move(self.board, old_position, new_position, self.current_player):
            self.skip_turn("illegal move")
            return False

        player = self.current_player
        self.finish_turn(f"Player {player} moved. Select a piece for a human turn.", player)
        return True

    def handle_click(self, mouse_position: Tuple[int, int]) -> None:
        if self.result.status != "ongoing":
            return
        if self.bot_functions[self.current_player] is not None:
            return

        x, y = mouse_position
        if not (BOARD_X <= x < BOARD_X + 5 * CELL_SIZE
                and BOARD_Y <= y < BOARD_Y + 5 * CELL_SIZE):
            return
        position = ((y - BOARD_Y) // CELL_SIZE, (x - BOARD_X) // CELL_SIZE)
        row, column = position
        if self.board[row][column] == self.current_player:
            self.selected = None if position == self.selected else position
            self.message = "Select your piece, then a highlighted square."
            self.message_color = TEXT_COLORS[self.current_player]
        elif self.selected is not None:
            self.attempt_move(self.selected, position)

    def update_bot(self) -> None:
        if (self.result.status != "ongoing"
                or pygame.time.get_ticks() < self.bot_due):
            return
        bot = self.bot_functions[self.current_player]
        if bot is None:
            return
        try:
            # Search receives a copy, so it cannot change the live board.
            references = bot(
                [row[:] for row in self.board],
                self.current_player,
                VISUALIZE_SEARCH_TREE,
            )
            if (not isinstance(references, tuple) or len(references) != 2
                    or not all(isinstance(value, str) for value in references)):
                raise ValueError("Expected a tuple of two square strings")
            old_reference, new_reference = references
            old_position = parse_position(old_reference)
            new_position = parse_position(new_reference)
        except Exception as error:
            # Student bot errors forfeit this turn, rather than stopping the game.
            print(f"Player {self.current_player} bot error: {type(error).__name__}: {error}")
            print(traceback.format_exc())
            self.skip_turn("bot error (see console)")
            return

        print(f"Player {self.current_player}: {old_reference} -> {new_reference}")
        self.attempt_move(old_position, new_position)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font,
             small_font: pygame.font.Font) -> None:
        screen.fill((245, 247, 251))

        def label(text: str, x: int, y: int, small: bool = False,
                  color: Tuple[int, int, int] = NEUTRAL_COLOR) -> None:
            screen.blit((small_font if small else font).render(
                text, True, color), (x, y))

        def player_list(title: str, players: List[int], y: int) -> None:
            label(title, 35, y, True, GAME_END_COLOR)
            x = 35 + small_font.size(title + " ")[0]
            if not players:
                label("none", x, y, True, GAME_END_COLOR)
            for player in players:
                text = f"P{player}  "
                label(text, x, y, True, TEXT_COLORS[player])
                x += small_font.size(text)[0]

        label("HALMA | 4 players", 35, 20)
        label(f"Turns: {self.move_count}/{MAXIMUM_MOVE_LIMIT} (includes skipped turns)",
              35, 58, True)

        if self.result.status == "winner":
            player_list("Winner:", self.result.winners, 88)
        elif self.result.status == "move_limit":
            label("Game ended: move limit reached", 35, 84, color=GAME_END_COLOR)
        else:
            actor = "bot" if self.bot_functions[self.current_player] is not None else "human"
            label(f"Player {self.current_player}'s turn ({actor})", 35, 84,
                  color=TEXT_COLORS[self.current_player])

        legal = self.destinations(self.selected) if self.selected is not None else []
        for index in range(5):
            label(chr(ord("A") + index), BOARD_X + index * CELL_SIZE + 36,
                  BOARD_Y - 27, True)
            label(str(index + 1), BOARD_X - 25,
                  BOARD_Y + index * CELL_SIZE + 34, True)

        for row in range(5):
            for column in range(5):
                position = (row, column)
                rect = pygame.Rect(BOARD_X + column * CELL_SIZE,
                                   BOARD_Y + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                background = (255, 255, 255) if (row + column) % 2 == 0 else (234, 239, 246)
                for player, cells in win_cells_all.items():
                    if position in cells:
                        background = ZONE_COLORS[player]
                pygame.draw.rect(screen, background, rect)
                pygame.draw.rect(screen, (179, 190, 205), rect, 1)
                if position == self.selected:
                    pygame.draw.rect(screen, (241, 185, 42), rect.inflate(-6, -6), 4)
                if position in legal:
                    pygame.draw.circle(screen, (65, 156, 110), rect.center, 10)
                piece = self.board[row][column]
                if piece:
                    pygame.draw.circle(screen, COLORS[piece], rect.center, 28)
                    number_color = NEUTRAL_COLOR if piece == 3 else (255, 255, 255)
                    number = font.render(str(piece), True, number_color)
                    screen.blit(number, number.get_rect(center=rect.center))

        if self.result.status == "move_limit":
            player_list("Tied players:", self.result.tied_players, 602)
            player_list("Lost for blocking:", self.result.losers, 628)
        elif self.result.status == "winner":
            label("Game ended: all three pieces reached the end zone.",
                  35, 602, True, GAME_END_COLOR)
            player_list("Other players:", self.result.losers, 628)
        else:
            label(self.message, 35, 602, True, self.message_color)
            label("Tinted squares are the matching player's end zone.", 35, 628, True)
        for player in range(1, 5):
            x = 35 + ((player - 1) % 2) * 285
            y = 666 + ((player - 1) // 2) * 27
            actor = "bot" if self.bot_functions[player] is not None else "human"
            label(f"Player {player}: {actor}", x, y, True, TEXT_COLORS[player])
        label("R: restart    |    Esc: quit", 35, 742, True)


def main(bot_functions: Optional[Dict[int, Optional[BotFunction]]] = None) -> None:
    pygame.init()
    try:
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Halma - 4 players")
        font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        game = HalmaGame(bot_functions)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    game.handle_click(event.pos)
            if not running:
                break
            game.update_bot()
            game.draw(screen, font, small_font)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
