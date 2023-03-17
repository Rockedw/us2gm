Given(/^that I arrive on the ffolio site$/) do
  visit '/'
end

Given(/^I register for a ffolio account$/) do
	click_link 'Sign up'
  fill_in 'Email', with: 'bla@bla.com'
  fill_in 'Password', with: '12345678'
  fill_in 'Password confirmation', with: '12345678'
  click_button 'Sign up'
end

Then(/^I should be on my profile$/) do
	@user = User.find_by_email 'bla@bla.com'
	expect(current_path).to eq "/users/#{@user.id}"
end

Then(/^I should be invited to fill in my profile details$/) do
  expect(page).to have_content 'Please fill in your profile details'
end