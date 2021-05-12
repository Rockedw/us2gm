class AddIsCorrectToMultipleChoiceAnswer < ActiveRecord::Migration[5.1]
  def self.up
    add_column :multiple_choice_answers, :is_correct, :boolean
  end

  def self.down
    remove_column :multiple_choice_answers, :is_correct
  end
end
