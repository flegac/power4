Feature: Detect endgame


  Scenario: Detect victory condition
    Given A power4 board position (state)
    When A player move creates 4 adjacent cells of the same color
    Then The state is terminal
    Then The score is a victory for the last playing player

  Scenario: Detect draw condition
    Given A power4 board position (state)
    When A player move fill the board (without victory)
    Then The state is terminal
    Then The score is a draw
