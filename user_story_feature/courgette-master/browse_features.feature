Feature: Browse features
  In order to find a specific feature among the mass of features of my project
  As a Product Owner
  I want to broswe the features with a treeview file explorer

  @javascript
  Scenario: Browswe folder and open a feature file source
    Given I enter the address of the features page
    Then it should display the list of features for my application in a treeview which looks like a file explorer

    When I click on the folder "business"
    And I click on the feature filename "visitor_transforms"

    Then it should open the "visitor_transforms.feature" file source

