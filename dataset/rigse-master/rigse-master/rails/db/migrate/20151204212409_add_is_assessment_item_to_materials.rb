class AddIsAssessmentItemToMaterials < ActiveRecord::Migration[5.1]
  def change
    add_column :investigations, :is_assessment_item, :boolean, :default =>false
    add_column :activities, :is_assessment_item, :boolean, :default =>false
    add_column :external_activities, :is_assessment_item, :boolean, :default =>false
  end
end
