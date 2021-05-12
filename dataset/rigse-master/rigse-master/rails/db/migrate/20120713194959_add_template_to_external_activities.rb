class AddTemplateToExternalActivities < ActiveRecord::Migration[5.1]
  def change
    add_column :external_activities, :template_id, :integer
    add_column :external_activities, :template_type, :string
  end
end
