class HomepageProjectCards < ActiveRecord::Migration[5.1]
  def change
    add_column :admin_projects, :project_card_image_url, :string
    add_column :admin_projects, :project_card_description, :string
  end
end
