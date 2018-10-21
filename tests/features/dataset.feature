Feature: Handle datasets


  Scenario: Create dataset from scratch
    When I create a dataset with some data
    Then The dataset object contains the original data

  Scenario: Split a dataset in multiple smaller datasets
    Given A dataset containing lots of samples
    When I split the dataset
    Then The sum of resulting dataset is equal to the original dataset
