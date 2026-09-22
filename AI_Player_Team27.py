from typing import Dict, List, Optional, Tuple, Callable
import pygame
import treelib as tr
from copy import deepcopy
import traceback
from graphviz import Source

from halma import (
    check_legal_move,
    move,
    win_cells_all,
    win_cells_1v1,
    GameResult,
)

def AI_Player_Team27(board: List[List[int]], player: int, visualize: bool = False, depth:int = 2) -> Tuple[str, str]:
    if len(board) != 5 or len(board[0]) != 5:
        raise ValueError("Board must be 5 by 5")
    if player not in [1, 2, 3, 4]:
        raise ValueError(f"Player {player} is not a valid player")
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    tree = generate_tree_2(board, player, depth)

    if visualize:
        tree.show()
        #tree.save2file("Team27_Tree.txt", line_type="ascii")
        tree.to_graphviz("Team27_Tree", graph="digraph")
        try:
            Source.from_file("Team27_Tree").render(format="png")
        except Exception as error:
            print(f"Could not convert to PNG: {error}")


    minimax_search_tree(tree, player, "root")

    best_move = None
    best_score = -float("inf")

    for child in tree.children("root"):
        # print(f"Child: {child.identifier}, Score: {child.data.get('score', 'N/A')}")
        score = child.data.get("score", -float("inf"))
        if score > best_score:
            best_score = score
            best_move = child.data.get("move")

    if best_move is None:
        return "", ""

    move_str = move_to_string(best_move)
    print(f"Best move for player {player}: {move_str} with score {best_score}")

    old_pos, new_pos = best_move
    old_ref: str = position_to_string(old_pos)
    new_ref: str = position_to_string(new_pos)
    print(f"Player {player} selected move: {old_ref} -> {new_ref}")
    return old_ref, new_ref


def position_to_string(position: Tuple[int, int]) -> str:
    return chr(ord("A") + position[1]) + str(position[0] + 1)


def move_to_string(move: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
    old_pos, new_pos = move
    return f"{position_to_string(old_pos)}->{position_to_string(new_pos)}"

# Scores how good a board is for that player
# by seeing how far removed they are from their
# target tile.
def evaluate_board(board: List[List[int]], player: int) -> int:
    total = 0
    for row in range(5):
        for col in range(5):
            if board[row][col] != player: continue

            if (row, col) in win_cells_all[player]:
                total += 50
                continue
            distances = [
                abs(row - target[0]) + abs(col - target[1])
                for target in win_cells_all[player]
            ]
            total += max(0, 25 - min(distances)) # board_length - min(distances)
    return total


# min max search with alpha beta pruning
def alpha_beta_pruning(
    board: List[List[int]],
    player: int,
    depth: int,
    alpha: float,
    beta: float,
    max_player: int,
) -> float:
    if depth == 0:
        return evaluate_board(board, max_player)

    moves = generate_possible_moves(board, player)
    if not moves:
        return evaluate_board(board, max_player)

    if player == max_player:
        best = -float("inf")
        for old_pos, new_pos in moves:
            board_copy = deepcopy(board)
            move(board_copy, old_pos, new_pos, player)
            score = alpha_beta_pruning(
                board_copy,
                (player % 4) + 1,
                depth - 1,
                alpha,
                beta,
                max_player,
            )
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    best = float("inf")
    for old_pos, new_pos in moves:
        board_copy = deepcopy(board)
        move(board_copy, old_pos, new_pos, player)
        score = alpha_beta_pruning(
            board_copy,
            (player % 4) + 1,
            depth - 1,
            alpha,
            beta,
            max_player,
        )
        best = min(best, score)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best

def minimax_search_tree(tree: tr.Tree, root_player: int, node_id: str = "root") -> float:
    node = tree.get_node(node_id)
    children = tree.children(node_id)

    if not children:
        score = alpha_beta_pruning(
            node.data["board"],
            node.data.get("player_to_move", root_player),
            0,
            -float("inf"),
            float("inf"),
            root_player,
        )
        node.data["score"] = score
        return score

    child_scores = []

    for child in children:
        score = minimax_search_tree(tree, root_player, child.identifier)
        child_scores.append(score)


    if node.data.get("player_to_move") == root_player:
        # maximising player
        score = max(child_scores)
    else:
        score = min(child_scores)

    node.data["score"] = score
    return score


def generate_tree_2(
        board: List[List[int]],
        player: int,
        depth: int,
        parent: str = "root",
        tree: Optional[tr.Tree] = None,
        seen: Optional[set] = None
) -> tr.Tree:
    if tree is None:
        tree = tr.Tree()
        tree.create_node(
            parent,
            parent,
            data={
                "board": board,
                "board_str": board_to_string(board),
                "player_to_move": player,
                "score": 0,
                "move": None,
                "move_str": None,  # for debugging
            },
        )

    if depth == 0:
        return tree
    if seen is None:
        seen = set()
    key = tuple(tuple(row) for row in board)
    if key in seen:
        return tree
    seen.add(key)

    possible_moves = generate_possible_moves(board, player)
    for old_pos, new_pos in possible_moves:
        board_copy = deepcopy(board)
        move(board_copy, old_pos, new_pos, player)
        move_id = f"{player}:{chr(ord("A") + old_pos[1]) + str(old_pos[0] + 1)}->{chr(ord("A") + new_pos[1]) + str(new_pos[0] + 1)}"
        if move_id in tree:
            continue

        tree.create_node(
            move_id,
            move_id,
            parent=parent,
            data={
                "board": board_copy,
                "board_str": board_to_string(board_copy),
                "player_to_move": (player % 4) + 1,
                "score": 0,
                "move": (old_pos, new_pos),
                "move_str": f"{position_to_string(old_pos)}->{position_to_string(new_pos)}", # for debugging
            },
        )
        generate_tree_2(board_copy, (player % 4) + 1, depth - 1, move_id, tree, seen)

    return tree

# recursivly generate a tree of possible moves for current and next players.
def generate_tree(board: List[List[int]], player: int, board_map, depth: int,move_tree:tr.Tree,parent = "root") -> None:
    if "root" not in move_tree: #first iteration (first call)
        move_tree.create_node(
            "root", "root", data={"board" : board,"board_str": board_to_string(board)}
        )
        board_map["root"] = board
    if depth == 0: #required depth reached, going up
        return None

    possible_moves = generate_possible_moves(board, player)

    # you have to call "generate_possible_moves(board, player)"
    for old_pos,new_pos in possible_moves: 
        board_copy = deepcopy(board)
        move(board_copy, old_pos, new_pos, player)
        move_id = f"{player}:{old_pos}->{new_pos}"
        if (move_id in move_tree) or (board_copy in board_map.values()):
            if move_tree.depth(move_id) > depth:
                #print(f"Move {move_id} already exists in tree and has been moved from depth {move_tree.depth(move_id)} to {depth}")
                #print(board_to_string(board_copy))
                move_tree.move_node(move_id, parent)
            continue

        data = {
            "board": board_copy,
            "board_str": board_to_string(board_copy), # str version of the board (for printing)
            "happiness": [0,0,0,0] # how much each player likes this board (slot 0 - player 1, slot 1 - player 2, etc)
        }
        move_tree.create_node(move_id, move_id, parent,data)
        board_map[move_id] = board_copy
        generate_tree(board_copy, player%4+1, board_map, depth-1, move_tree, move_id)

# generates all possible moves for given player and board.
def generate_possible_moves(board: List[List[int]],player) -> List[Tuple[int,int]]:
    possible_moves = []
    for row in range(5):
        for column in range(5):
            if board[row][column] != player : continue
            old_pos = (row,column)

            for newRow in range(-2,2):
                for newColumn in range(-2,2):
                    new_pos = (row+newRow,column+newColumn)
                    if new_pos[0] < 0 or new_pos[0] > 4 or new_pos[1] < 0 or new_pos[1] > 4: continue
                    if not check_legal_move(board,old_pos,new_pos): continue
                    possible_moves.append((old_pos,new_pos))
    return possible_moves

# serializes the board into an easy printable string.
def board_to_string(board: List[List[int]]) -> str:
    returnable:str = ""
    for row in board:
        for cell in row:
            returnable += str(cell).replace("0",".")
        returnable += "\n"
    return returnable

def get_ancestors(node_list: List[tr.Node], tree: tr.Tree) -> List[tr.Node]:
    ancestors = []
    try:
        for node in node_list:
            id = tree.ancestor(node.identifier)
            if id is None: continue
            node = tree.get_node(id)
            if node is None: continue
            ancestors.append(node)
        return list(set(ancestors))
    except:
        return []