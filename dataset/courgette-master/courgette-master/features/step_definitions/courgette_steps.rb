Given /^I enter the address of the features page$/ do
  visit '/features'
end

Given /^I enter the address of the features page, e\.g\. http:\/\/staging\.myfantasticapp\.com\/features$/ do
  visit '/features'
end

Then /^I should see the source code of the first feature of my application$/ do
  Then %{I should see "feature with the same filename as the one in features"}
end

Then /^I should see a table where a row contains the cells? "([^\"]*)"$/ do |cells|
  response.should have_tag('table') do |table|
    table.should have_tag('tr') do |tr|
      cells.split(/\s*,\s*/).each do |cell|
        tr.should have_tag('td,th', :text => /#{Regexp.escape(cell)}/)
      end
    end
  end
end

Then /^I should see a table with the headers "([^\"]*)"$/ do |headers|
  response.should have_tag('table') do |table|
    headers.split(/\s*,\s*/).each do |header|
      table.should have_tag('th', :text => /#{Regexp.escape(header)}/)
    end
  end
end

When /^I click on the folder "([^\"]*)"$/ do |folder_name|
  click_link folder_name
end

Then /^it should display the list of features for my application in a treeview which looks like a file explorer$/ do
  # no op, purely descriptive step
end

When /^I click on the feature filename "([^"]*)"$/ do |filename|
  click_link filename
end

Then /^it should open the "visitor_transforms.feature" file source$/ do
  Then %{I should see "Feature: Visitor can transform into a tamagochi"}
end


