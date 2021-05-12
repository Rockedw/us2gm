class AddEnableTextFeedbackToEmbeddableMetadata < ActiveRecord::Migration[5.1]
  def change
    add_column :portal_offering_embeddable_metadata, :enable_text_feedback, :boolean, :default => true
  end
end
