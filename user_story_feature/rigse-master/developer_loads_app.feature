Feature: A developer loads home page without initializing data in database
  As a Developer
  I want to load the app before I've finished the setup
  So that I can see what I need to do next
  
  @no-seeds
  Scenario: A developer looks at the home page
    Given I am on the home page
    Then I should see "You need to create a settings object"

