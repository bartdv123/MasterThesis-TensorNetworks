
mutable struct Game
state::Int
function Game(state)
    new(state)
    
end
end


function change_state(game::Game, ns::Int)
    game.state = ns
end

current_state(game::Game) = game.state




function step(game::Game)
    state = copy(current_state(game))
    print(state)

    change_state(game, 5)

    print(state)

end
game = Game(2)
step(game)