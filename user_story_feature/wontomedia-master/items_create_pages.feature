Feature:  Create and view new individual items through non-Ajax pages
  In order to create a wontology,
  As a contributor,
  I want to create and view items.

  Scenario: Create new item
    Given I am on the new items page
    And I fill in "Name" with "MyCategory"
    And I fill in "Title" with "A subcategory"
    And I fill in "Description" with "The root category in the C topic"
    And I select "Category" from "Type"
    When I press "Create"
    Then I should see "successfully created"
    And I should see "MyCategory"
    And I should see "A subcategory"
    And I should see "The root category in the C topic"


  Scenario: System should prevent entry of invalid item names
    Given I am on the new items page
    And I fill in "Name" with "0bad"
    And I fill in "Title" with "A good title /\?"
    And I fill in "Description" with "0 And a (good) description, too."
    And I select "Individual" from "Type"
    When I press "Create"
    Then I should see "error"
    And I fill in "Name" with "bad too"
    When I press "Create"
    Then I should see "Could not create"
    And I fill in "Name" with "BAD>bad"
    When I press "Create"
    Then I should see "Could not create"


  Scenario: I can't enter two items with same name
    Given I am on the new items page
    And I fill in "Name" with "original"
    And I fill in "Title" with "Original Item"
    And I fill in "Description" with "description"
    And I select "Category" from "Type"
    When I press "Create"
    Then I should see "successfully created"
    Given I am on the new items page
    And I fill in "Name" with "original"
    And I fill in "Title" with "Second Item"
    And I fill in "Description" with "Actually second item, but bad name"
    And I select "Individual" from "Type"
    When I press "Create"
    Then I should see "error"


  @not_for_selenium
  Scenario: System should prevent entry of invalid item titles
    Given I am on the new items page
    And I fill in "Name" with "goodName"
    And I fill in "Title" with "Bad title\012has two lines"
    And I fill in "Description" with "good"
    And I select "Individual" from "Type"
    When I press "Create"
    Then I should see "error"


# WontoMedia - a wontology web application
# Copyright (C) 2011 - Glen E. Ivey
#    www.wontology.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License version
# 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program in the file COPYING and/or LICENSE.  If not,
# see <http://www.gnu.org/licenses/>.
