Feature: feature url
  In order to my customers to save time when browsing cuke features in Courgette
  As a developper
  I want to send them directly url such as /features/my_great_feature

  @javascript
  Scenario: Open a feature in Courgette and see the direct link appear in the header so I can copy it and send it by email
    Given I enter the address of the features page
    When I follow "direct link to send by email"
    Then I should see the source code of the first feature of my application

