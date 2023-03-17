Given(/^I specify that I wish to upload text$/) do

  click_button 'Add To My Collection'
  click_link 'Add Writing'
end

When(/^I enter text$/) do
  find(:css, "#texteditor").set "The apparition of the faces in the crowd <br> Petals on a wet black bough"
end

When(/^I upload a thumbnail, add a title, genres and a short description$/) do
  click_link 'Add Details'
  attach_file 'Thumbnail', Rails.root.join('features/images/art.jpg')
  fill_in 'Title', with: 'In A Station of the Metro'
  fill_in 'work_genre_names', with: 'Modernism, Imagism, Vorticism'
  fill_in 'work_caption', with: 'an intellectual and emotional complex in an instant of time.'
end

Then(/^I should see the work's thumbnail, title, short description and genres in the default group$/) do
  expect(page).to have_css 'img.thumb'
  click_link 'Add Details'
  expect(page).to have_content 'In A Station of the Metro'
  expect(page).to have_content 'Modernism, Imagism, Vorticism'
  expect(page).to have_content 'an intellectual and emotional complex in an instant of time.'
end


Then(/^I should see the content of the work$/) do
  expect(page).to have_content "The apparition of the faces in the crowd \n Petals on a wet black bough"
end